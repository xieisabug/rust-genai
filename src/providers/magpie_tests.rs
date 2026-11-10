use super::*;
use std::path::Path;

#[test]
fn test_hello_requires_magpie_name() {
    assert_eq!(hello_name(r#"{"name":"magpie","version":"1"}"#).unwrap(), "magpie");
    assert!(hello_name(r#"{"name":"other"}"#).unwrap() != "magpie");
    assert!(hello_name("not-json").is_err());
}

#[test]
fn test_settings_path_prefers_xdg_then_home() {
    assert_eq!(
        settings_path(Some(Path::new("/tmp/xdg")), Some(Path::new("/home/me"))).unwrap(),
        Path::new("/tmp/xdg/magpie/settings.json")
    );
    assert_eq!(
        settings_path(Some(Path::new("")), Some(Path::new("/home/me"))).unwrap(),
        Path::new("/home/me/.config/magpie/settings.json")
    );
    assert!(settings_path(None, None).is_none());
}

#[test]
fn test_port_from_settings_reads_positive_port_only() {
    assert_eq!(port_from_settings(r#"{"port":3426,"theme":"dark"}"#), Some(3426));
    assert_eq!(port_from_settings("{}"), None);
    assert_eq!(port_from_settings(r#"{"port":0}"#), None);
    assert_eq!(port_from_settings("not json"), None);
}

#[test]
fn test_origin_strips_client_suffix_and_keeps_proxy_prefix() {
    assert_eq!(origin_from_endpoint("http://127.0.0.1:3425/v1/").unwrap(), "http://127.0.0.1:3425");
    assert_eq!(origin_from_endpoint("http://127.0.0.1:3425/v1beta/").unwrap(), "http://127.0.0.1:3425");
    assert_eq!(origin_from_endpoint("http://127.0.0.1:3425/").unwrap(), "http://127.0.0.1:3425");
    assert_eq!(origin_from_endpoint("https://gw.example.com/magpie/v1/").unwrap(), "https://gw.example.com/magpie");
    assert!(origin_is_loopback("http://127.0.0.1:3425"));
    assert!(origin_is_loopback("http://localhost:3425"));
    assert!(!origin_is_loopback("https://gw.example.com"));
    assert!(origin_is_loopback("http://[::1]:3425"));
    for invalid in ["ftp://host", "https://user:secret@host", "https://host?key=secret", "https://host#fragment", "bad", "http:///path", "https:host", "http://host\\path", "http://local\nhost"] {
        assert!(Gateway::new(Some(invalid), "test-key").is_err(), "{invalid}");
    }
}

#[test]
fn test_invalid_protocol_and_header_fail_before_transport() {
    let gateway = Gateway::new(Some("http://127.0.0.1:1"), "test-key").unwrap();
    assert!(gateway.client(Some("future"), WebConfig::default(), "Test/1", None).unwrap_err().contains("future"));
    assert!(gateway.client(None, WebConfig::default(), "bad\nagent", None).unwrap_err().contains("user-agent"));
    assert!(gateway.client(None, WebConfig::default(), "Test/1", Some("bad\nsession")).unwrap_err().contains("X-Magpie-Session"));
    assert!(Gateway::new(Some("http://localhost"), " ").is_err());
}

// Real loopback HTTP catches protocol paths, header precedence, proxy bypass
// and preservation of routing-group model IDs across all adapters.
async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        let n = socket.read(&mut chunk).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length: usize = headers.lines().find_map(|line| line.strip_prefix("content-length: "))
                .map(|v| v.parse().unwrap()).unwrap_or(0);
            if bytes.len() >= end + 4 + length { break; }
        }
    }
    String::from_utf8(bytes).unwrap()
}

#[tokio::test]
async fn test_protocol_http_paths_auth_and_session_preserve_gateway_model_id() {
    use crate::chat::{ChatMessage, ChatRequest};
    use tokio::io::AsyncWriteExt;
    for (mode, path) in [
        ("chat_completions", "/v1/chat/completions"),
        ("responses", "/v1/responses"),
        ("anthropic_messages", "/v1/messages"),
        ("google_gemini", "/v1beta/models/auto/test-model:generateContent"),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway = Gateway::new(Some(&format!("http://{}/v1", listener.local_addr().unwrap())), "test-key").unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            socket.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
            request
        });
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer wrong"));
        headers.insert("x-custom", HeaderValue::from_static("kept"));
        let config = WebConfig::default().with_default_headers(headers)
            .with_timeout(std::time::Duration::from_secs(3))
            .with_all_proxy_url("http://127.0.0.1:1").unwrap();
        let client = gateway.client(Some(mode), config, "Example/1", Some("session-uuid")).unwrap();
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        assert!(client.exec_chat("auto/test-model", request, None).await.is_err());
        let request = tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
        assert!(request.starts_with(&format!("POST {path}")), "{mode}: {request}");
        let headers = request.split("\r\n\r\n").next().unwrap().to_ascii_lowercase();
        assert!(headers.contains("authorization: bearer test-key"), "{headers}");
        assert!(headers.contains("user-agent: example/1"));
        assert!(headers.contains("x-magpie-session: session-uuid"));
        assert!(headers.contains("x-custom: kept"));
        if mode == "anthropic_messages" { assert!(headers.contains("x-api-key: test-key")); }
        if mode == "google_gemini" { assert!(headers.contains("x-goog-api-key: test-key")); }
        if mode != "google_gemini" {
            let body: serde_json::Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
            assert_eq!(body["model"], "auto/test-model", "{mode}");
        }
    }
}

#[tokio::test]
async fn test_catalog_http_verifies_gateway_and_reports_failures_with_url() {
    use tokio::io::AsyncWriteExt;
    for (hello, status, catalog, expected) in [
        (r#"{"name":"magpie"}"#, "200 OK", r#"{"data":[{"id":"auto/model","native_endpoints":["/v1/messages"]}]}"#, ""),
        (r#"{"name":"other"}"#, "200 OK", "{}", "不是 magpie"),
        ("not-json", "200 OK", "{}", "不是 JSON"),
        (r#"{"name":"magpie"}"#, "503 Unavailable", "gateway unavailable", "503"),
        (r#"{"name":"magpie"}"#, "200 OK", "not-json", "模型目录无法解析"),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let gateway = Gateway::new(Some(&origin), "local-test-key").unwrap();
        let count = if hello == r#"{"name":"magpie"}"# { 2 } else { 1 };
        let server = tokio::spawn(async move {
            for index in 0..count {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_request(&mut socket).await;
                let path = if index == 0 { "/api/hello" } else { "/v1/models" };
                assert!(request.starts_with(&format!("GET {path} HTTP/1.1")));
                assert!(request.to_ascii_lowercase().contains("authorization: bearer local-test-key"));
                let (body, status) = if index == 0 { (hello, "200 OK") } else { (catalog, status) };
                socket.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        });
        let result = gateway.catalog(WebConfig::default().with_timeout(std::time::Duration::from_secs(3)), "Example/1").await;
        if expected.is_empty() {
            let models = result.unwrap();
            assert_eq!(models[0].id, "auto/model");
            assert_eq!(models[0].available_modes[0], "anthropic_messages");
        } else {
            let error = result.unwrap_err();
            assert!(error.contains(expected), "{error}");
            assert!(error.contains(&origin), "{error}");
        }
        tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
    }
}

#[test]
fn test_routes_hit_magpie_paths() {
    let origin = "http://127.0.0.1:3425";
    let (_, chat) = route(Some("chat_completions"), origin).unwrap();
    assert_eq!(reqwest::Url::parse(&chat).unwrap().join("chat/completions").unwrap().as_str(),
        "http://127.0.0.1:3425/v1/chat/completions");
    let (_, responses) = route(Some("responses"), origin).unwrap();
    assert_eq!(reqwest::Url::parse(&responses).unwrap().join("responses").unwrap().as_str(),
        "http://127.0.0.1:3425/v1/responses");
    let (anthropic, base) = route(Some("anthropic_messages"), origin).unwrap();
    assert_eq!(anthropic, AdapterKind::Anthropic);
    assert_eq!(format!("{base}messages"), "http://127.0.0.1:3425/v1/messages");
    let (gemini, base) = route(Some("google_gemini"), origin).unwrap();
    assert_eq!(gemini, AdapterKind::Gemini);
    assert_eq!(format!("{base}models/openai/gpt-5:generateContent"),
        "http://127.0.0.1:3425/v1beta/models/openai/gpt-5:generateContent");
    assert!(route(Some("future"), origin).is_err());
}

#[test]
fn test_catalog_keeps_chat_models_and_native_protocol_first() {
    let body = r#"{"object":"list","data":[
        {"id":"anthropic/claude","display_name":"Claude","owned_by":"anthropic",
         "native_endpoints":["/v1/messages"],"modalities":{"input":["text","image"]}},
        {"id":"openai/gpt","display_name":"","owned_by":"openai","kind":"chat"},
        {"id":"draw","kind":"image","display_name":"Draw"},
        {"id":"","display_name":"blank"}
    ]}"#;
    let models = parse_catalog(body).unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name, "Claude · anthropic");
    assert_eq!(models[0].id, "anthropic/claude");
    assert!(models[0].vision_support);
    assert_eq!(models[0].available_modes[0], "anthropic_messages");
    assert_eq!(models[0].available_modes[0], "anthropic_messages");
    assert_eq!(models[0].available_modes.len(), 4);
    assert_eq!(models[1].name, "openai/gpt");
    assert_eq!(models[1].available_modes[0], "chat_completions");
    assert_eq!(models[1].owner, "openai");
}

/// 同名模型优先使用网关提供的来源标签，路由组与直连提供商保持独立。
#[test]
fn test_catalog_distinguishes_providers_and_routing_groups_with_labels() {
    let body = r#"{"data":[
        {"id":"auto/claude","display_name":"Claude","magpie_label":"Claude · routing group","owned_by":"claude-mine"},
        {"id":"claude-mine/claude","display_name":"Claude","magpie_label":"Claude · 我的订阅","owned_by":"claude-mine"},
        {"id":"other/claude","display_name":"Claude","magpie_label":"Claude · 另一个渠道","owned_by":"other"}
    ]}"#;
    let models = parse_catalog(body).unwrap();
    assert_eq!(models.iter().map(|model| model.name.as_str()).collect::<Vec<_>>(),
        ["Claude · routing group", "Claude · 我的订阅", "Claude · 另一个渠道"]);
    assert_eq!(models.iter().map(|model| model.id.as_str()).collect::<Vec<_>>(),
        ["auto/claude", "claude-mine/claude", "other/claude"]);
    assert_eq!(models[0].available_modes[0], "chat_completions");
    assert_eq!(models[1].available_modes[0], "chat_completions");
    assert_eq!(models[2].available_modes[0], "chat_completions");
}

/// 旧版目录缺少来源标签时使用路由前缀；名称缺失时保留完整 ID。
#[test]
fn test_catalog_names_without_labels_keep_route_identity() {
    let body = r#"{"data":[
        {"id":"auto/claude","display_name":"Claude","owned_by":"mine"},
        {"id":"mine/claude","display_name":"Claude","magpie_label":"  ","owned_by":"mine"},
        {"id":"other/claude","display_name":"Claude"},
        {"id":"mine/gpt","display_name":"  ","owned_by":"mine"},
        {"id":"mine/plain","display_name":"mine/plain","owned_by":"mine"},
        {"id":"group","display_name":"My group"},
        {"id":"legacy","display_name":"Legacy","owned_by":"vendor"}
    ]}"#;
    let models = parse_catalog(body).unwrap();
    assert_eq!(models.iter().map(|model| model.name.as_str()).collect::<Vec<_>>(),
        ["Claude · auto", "Claude · mine", "Claude · other", "mine/gpt", "mine/plain", "My group", "Legacy · vendor"]);
}

/// 提供商名称被隐藏或多个渠道使用相同标签时，仍可按完整模型 ID 区分。
#[test]
fn test_catalog_duplicate_labels_include_full_model_ids() {
    let body = r#"{"data":[
        {"id":"one/claude","display_name":"Claude","magpie_label":"Claude"},
        {"id":"two/claude","display_name":"Claude","magpie_label":"Claude"},
        {"id":"one/gpt","display_name":"GPT","magpie_label":"GPT · One"}
    ]}"#;
    let models = parse_catalog(body).unwrap();
    assert_eq!(models.iter().map(|model| model.name.as_str()).collect::<Vec<_>>(),
        ["Claude (one/claude)", "Claude (two/claude)", "GPT · One"]);
}

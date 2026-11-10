//! Magpie gateway integration for all four text protocols.
//!
//! Gateway discovery, catalog metadata and transport live here. Applications
//! supply their own API key (e.g. `magpie-myapp` locally), User-Agent, session
//! identifier and saved protocol preference. Model IDs are sent unchanged.

use crate::adapter::AdapterKind;
use crate::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use crate::{Client, ModelIden, ServiceTarget, WebConfig};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const TEXT_PROTOCOLS: [&str; 4] = ["chat_completions", "responses", "anthropic_messages", "google_gemini"];
pub const DEFAULT_PORT: u16 = 3425;

pub fn mode(value: Option<&str>) -> Option<&'static str> {
    let value = value?;
    TEXT_PROTOCOLS.into_iter().find(|mode| *mode == value)
}

/// `$XDG_CONFIG_HOME/magpie/settings.json`, otherwise `~/.config/magpie/settings.json`.
pub fn settings_path(xdg: Option<&Path>, home: Option<&Path>) -> Option<PathBuf> {
    if let Some(xdg) = xdg.filter(|path| !path.as_os_str().is_empty()) {
        return Some(xdg.join("magpie").join("settings.json"));
    }
    Some(home?.join(".config").join("magpie").join("settings.json"))
}

pub fn port_from_settings(text: &str) -> Option<u16> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let port = value.get("port")?.as_u64()?;
    u16::try_from(port).ok().filter(|port| *port > 0)
}

pub fn discovered_port() -> Option<u16> {
    let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
    let home = dirs::home_dir();
    let path = settings_path(xdg.as_deref(), home.as_deref())?;
    port_from_settings(&std::fs::read_to_string(path).ok()?)
}

pub fn default_origin() -> String {
    format!("http://127.0.0.1:{}", discovered_port().unwrap_or(DEFAULT_PORT))
}

/// The gateway origin. An OpenAI `/v1` or Gemini `/v1beta` suffix is the client base, not a path prefix.
pub fn origin_from_endpoint(endpoint: &str) -> Result<String, String> {
    let invalid = || "Magpie endpoint must be an HTTP(S) URL with a host, without whitespace, backslashes, credentials, query or fragment".to_string();
    let (_, authority) = endpoint.split_once("://").ok_or_else(invalid)?;
    if authority.is_empty() || authority.starts_with('/') || endpoint.contains('\\')
        || endpoint.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(invalid());
    }
    let mut url = reqwest::Url::parse(endpoint).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none()
        || !url.username().is_empty() || url.password().is_some()
        || url.query().is_some() || url.fragment().is_some()
    {
        return Err(invalid());
    }
    let mut path = url.path().trim_end_matches('/').to_string();
    for suffix in ["/v1beta", "/v1"] {
        if let Some(stripped) = path.strip_suffix(suffix) {
            path = stripped.to_string();
            break;
        }
    }
    if path.is_empty() {
        path = "/".into();
    }
    url.set_path(&path);
    let mut origin = url.to_string();
    if origin.ends_with('/') {
        origin.pop();
    }
    Ok(origin)
}

pub fn origin_is_loopback(origin: &str) -> bool {
    reqwest::Url::parse(origin).ok().and_then(|url| url.host_str().map(|host| {
        host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1" || host == "[::1]"
    })).unwrap_or(false)
}

pub fn route(value: Option<&str>, origin: &str) -> Result<(AdapterKind, String), String> {
    let mode = match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => "chat_completions",
        Some(raw) => mode(Some(raw)).ok_or_else(|| format!(
            "Magpie 协议 {raw} 未受支持。可用协议：Chat Completions、Responses、Anthropic Messages、Google Gemini"))?,
    };
    let origin = origin.trim_end_matches('/');
    let (adapter, suffix) = match mode {
        "responses" => (AdapterKind::OpenAIResp, "/v1/"),
        "anthropic_messages" => (AdapterKind::Anthropic, "/v1/"),
        "google_gemini" => (AdapterKind::Gemini, "/v1beta/"),
        _ => (AdapterKind::OpenAI, "/v1/"),
    };
    Ok((adapter, format!("{origin}{suffix}")))
}

/// A chat model advertised by Magpie, independent of application persistence.
#[derive(Debug, Clone)]
pub struct ChatModel {
    pub id: String,
    pub name: String,
    pub owner: String,
    pub vision_support: bool,
    pub audio_support: bool,
    pub video_support: bool,
    /// Native protocols first, followed by gateway-translated protocols.
    pub available_modes: Vec<String>,
}

fn protocol_of(path: &str) -> Option<&'static str> {
    match path {
        "/v1/chat/completions" => Some("chat_completions"),
        "/v1/responses" => Some("responses"),
        "/v1/messages" => Some("anthropic_messages"),
        path if path.starts_with("/v1beta/") => Some("google_gemini"),
        _ => None,
    }
}

fn modes_for(native: &[String]) -> Vec<String> {
    let mut modes = Vec::new();
    for path in native {
        if let Some(mode) = protocol_of(path) {
            if !modes.iter().any(|existing| existing == mode) {
                modes.push(mode.to_string());
            }
        }
    }
    for protocol in TEXT_PROTOCOLS {
        if !modes.iter().any(|existing| existing == protocol) {
            modes.push((*protocol).to_string());
        }
    }
    modes
}

#[derive(Deserialize)]
struct Catalog { data: Vec<CatalogModel> }

#[derive(Deserialize)]
struct CatalogModel {
    id: String,
    #[serde(default)]
    display_name: String,
    #[serde(default)]
    magpie_label: String,
    #[serde(default)]
    owned_by: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    native_endpoints: Vec<String>,
    #[serde(default)]
    modalities: Modalities,
}

#[derive(Default, Deserialize)]
struct Modalities {
    #[serde(default)]
    input: Vec<String>,
}

impl CatalogModel {
    fn list_name(&self) -> String {
        if !self.magpie_label.trim().is_empty() {
            return self.magpie_label.clone();
        }
        let name = self.display_name.trim();
        if name.is_empty() || name == self.id {
            return self.id.clone();
        }
        // A routing group's owned_by may name its current upstream. The ID
        // prefix identifies the route the user actually selects (e.g. auto).
        let source = self.id.split_once('/').map(|(source, _)| source.trim())
            .filter(|source| !source.is_empty()).unwrap_or(self.owned_by.trim());
        if source.is_empty() { name.to_string() } else { format!("{name} · {source}") }
    }

    fn is_chat(&self) -> bool {
        let kind = self.kind.trim();
        kind.is_empty() || kind == "chat" || kind == "model"
    }

    fn into_model(self) -> ChatModel {
        let modes = modes_for(&self.native_endpoints);
        let supports = |kind: &str| self.modalities.input.iter().any(|value| value.split('/').next() == Some(kind));
        let name = self.list_name();
        ChatModel {
            vision_support: supports("image"), audio_support: supports("audio"), video_support: supports("video"),
            name, id: self.id, owner: self.owned_by, available_modes: modes,
        }
    }
}

pub fn parse_catalog(body: &str) -> Result<Vec<ChatModel>, String> {
    let catalog: Catalog = serde_json::from_str(body)
        .map_err(|e| format!("Magpie 模型目录无法解析: {e}"))?;
    let mut models: Vec<_> = catalog.data.into_iter().filter(|model| model.is_chat() && !model.id.trim().is_empty()).map(|model| {
        model.into_model()
    }).collect();
    let mut name_counts = HashMap::new();
    for model in &models {
        *name_counts.entry(model.name.clone()).or_insert(0) += 1;
    }
    for model in &mut models {
        if name_counts[&model.name] > 1 {
            model.name = format!("{} ({})", model.name, model.id);
        }
    }
    Ok(models)
}

fn hello_name(body: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| format!("不是 JSON: {e}"))?;
    value.get("name").and_then(|name| name.as_str()).filter(|name| !name.is_empty())
        .map(str::to_string).ok_or_else(|| "缺少 name".to_string())
}

async fn confirm_gateway(client: &reqwest::Client, origin: &str) -> Result<(), String> {
    let url = format!("{}/api/hello", origin.trim_end_matches('/'));
    let response = client.get(&url).send().await.map_err(|e| format!(
        "Magpie 网关 {url} 请求失败: {e}。请确认 Magpie 已启动（本机默认 http://127.0.0.1:{DEFAULT_PORT}）"))?;
    let status = response.status();
    let body = response.text().await.map_err(|e| format!("Magpie 网关 {url} 读取响应失败: {e}"))?;
    let detail: String = body.chars().take(300).collect();
    if !status.is_success() {
        return Err(format!("Magpie 网关 {url} 返回 {status}: {detail}"));
    }
    let name = hello_name(&body).map_err(|e| format!("地址 {origin} 的 /api/hello 不是 Magpie：{e}。响应：{detail}"))?;
    if name != "magpie" {
        return Err(format!("地址 {origin} 的 /api/hello 返回 name={name}，不是 magpie"));
    }
    Ok(())
}

/// A connection to a specific Magpie gateway. No application identity is assumed.
#[derive(Clone)]
pub struct Gateway {
    origin: String,
    api_key: String,
}

impl Gateway {
    /// Empty endpoints discover the local installation's port.
    /// Construct outside an async task when using local settings discovery.
    pub fn new(endpoint: Option<&str>, api_key: &str) -> Result<Self, String> {
        let origin = match endpoint.map(str::trim).filter(|value| !value.is_empty()) {
            Some(endpoint) => origin_from_endpoint(endpoint)?,
            None => default_origin(),
        };
        let api_key = api_key.trim();
        if api_key.is_empty() {
            return Err("Magpie requires a gateway key or a local magpie-<application> key".into());
        }
        Ok(Self { origin, api_key: api_key.to_string() })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    fn http_client(&self, mut config: WebConfig, user_agent: &str, session: Option<&str>) -> Result<reqwest::Client, String> {
        let mut headers = config.default_headers.take().unwrap_or_default();
        Self::identity_headers(&mut headers, user_agent, session, &self.api_key)?;
        config.default_headers = Some(headers);
        let local = origin_is_loopback(&self.origin);
        if local {
            config.proxy = None;
        }
        let mut builder = config.apply_to_builder(reqwest::Client::builder());
        if local {
            // Also disable environment/system proxies, not just explicit ones.
            builder = builder.no_proxy();
        }
        builder.build().map_err(|e| format!("Magpie 网关 {} 创建客户端失败: {e}", self.origin))
    }

    fn identity_headers(headers: &mut HeaderMap, user_agent: &str, session: Option<&str>, key: &str) -> Result<(), String> {
        let authorization = format!("Bearer {key}");
        for (name, value) in [("user-agent", user_agent), ("authorization", authorization.as_str())] {
            headers.insert(name, HeaderValue::from_str(value).map_err(|e| format!("Magpie {name} 请求头无效: {e}"))?);
        }
        // A caller may omit session identity, including for catalog-only clients.
        headers.remove("x-magpie-session");
        if let Some(session) = session {
            headers.insert("x-magpie-session", HeaderValue::from_str(session)
                .map_err(|e| format!("Magpie X-Magpie-Session 请求头无效: {e}"))?);
        }
        Ok(())
    }

    /// Create a genai client bound to the chosen protocol. Existing adapters
    /// handle request/response translation and streaming; no model-name inference
    /// or namespace stripping is applied to gateway model IDs.
    pub fn client(&self, request_mode: Option<&str>, config: WebConfig, user_agent: &str, session: Option<&str>) -> Result<Client, String> {
        let (kind, base) = route(request_mode, &self.origin)?;
        let key = self.api_key.clone();
        let resolver = ServiceTargetResolver::from_resolver_fn(move |target: ServiceTarget| {
            Ok(ServiceTarget {
                endpoint: Endpoint::from_owned(base.clone()),
                auth: AuthData::from_single(key.clone()),
                model: ModelIden::new(kind, target.model.model_name),
            })
        });
        Client::builder()
            .with_adapter_kind(kind)
            .with_service_target_resolver(resolver)
            .with_reqwest(self.http_client(config, user_agent, session)?)
            .build()
            .map_err(|e| format!("Magpie 网关 {} 创建 genai 客户端失败: {e}", self.origin))
    }

    /// Verify the gateway before retrieving its chat model catalog.
    pub async fn catalog(&self, config: WebConfig, user_agent: &str) -> Result<Vec<ChatModel>, String> {
        let client = self.http_client(config, user_agent, None)?;
        confirm_gateway(&client, &self.origin).await?;
        let url = format!("{}/v1/models", self.origin);
        let response = client.get(&url).bearer_auth(&self.api_key).send().await
            .map_err(|e| format!("Magpie 模型目录 {url} 请求失败: {e}"))?;
        let status = response.status();
        let body = response.text().await.map_err(|e| format!("Magpie 模型目录 {url} 读取响应失败: {e}"))?;
        if !status.is_success() {
            let detail: String = body.chars().take(500).collect();
            return Err(format!("Magpie 模型目录 {url} 返回 {status}: {detail}"));
        }
        parse_catalog(&body).map_err(|e| format!("{e}（{url}）"))
    }
}

#[cfg(test)]
#[path = "magpie_tests.rs"]
mod tests;

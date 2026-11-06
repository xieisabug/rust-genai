//! Local-only migration regressions: no provider credentials or remote requests.
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{
	BinarySource, ChatOptions, ChatRequest, ChatResponse, ChatStreamEvent, ContentPart, MessageContent, StreamEnd,
	ToolCall, ToolResponse,
};
use genai::resolver::{AuthData, Endpoint, ProviderConfig, ServiceTargetResolver};
use genai::{Client, Model, ModelIden, ServiceTarget};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn serve(body: String, content_type: &'static str) -> (Endpoint, tokio::task::JoinHandle<String>) {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let endpoint = Endpoint::from_owned(format!("http://{}/", listener.local_addr().unwrap()));
	let task = tokio::spawn(async move {
		let (mut socket, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
			.await
			.unwrap()
			.unwrap();
		let mut request = Vec::new();
		loop {
			let mut buf = [0; 8192];
			let n = socket.read(&mut buf).await.unwrap();
			assert!(n > 0);
			request.extend_from_slice(&buf[..n]);
			if let Some(end) = request.windows(4).position(|s| s == b"\r\n\r\n") {
				let headers = String::from_utf8_lossy(&request[..end]);
				let length = headers
					.lines()
					.find_map(|line| {
						let (name, value) = line.split_once(':')?;
						name.eq_ignore_ascii_case("content-length")
							.then(|| value.trim().parse::<usize>().unwrap())
					})
					.unwrap_or(0);
				if request.len() >= end + 4 + length {
					break;
				}
			}
		}
		let response = format!(
			"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
			body.len()
		);
		socket.write_all(response.as_bytes()).await.unwrap();
		String::from_utf8(request).unwrap()
	});
	(endpoint, task)
}

fn client() -> Client {
	Client::builder()
		.with_reqwest(
			reqwest::Client::builder()
				.no_proxy()
				.timeout(Duration::from_secs(10))
				.build()
				.unwrap(),
		)
		.build()
		.unwrap()
}

async fn stream(events: Vec<Value>, capture: bool) -> (StreamEnd, String) {
	let body = events.iter().map(|e| format!("data: {e}\n\n")).collect();
	let (endpoint, server) = serve(body, "text/event-stream").await;
	let target = ServiceTarget {
		endpoint,
		auth: AuthData::from_single("fixture"),
		model: ModelIden::new(AdapterKind::OpenAIResp, "gpt-5"),
	};
	let options = ChatOptions::default().with_capture_content(capture);
	let mut response = client()
		.exec_chat_stream(target, ChatRequest::from_user("fixture"), Some(&options))
		.await
		.unwrap()
		.stream;
	let mut end = None;
	let mut text = String::new();
	while let Some(event) = response.next().await {
		match event.unwrap() {
			ChatStreamEvent::End(value) => {
				assert!(end.is_none());
				end = Some(value);
			}
			ChatStreamEvent::Chunk(chunk) => text.push_str(&chunk.content),
			_ => (),
		}
	}
	server.await.unwrap();
	(end.expect("terminal event"), text)
}

fn partial(index: usize, bytes: &str) -> Value {
	json!({"type":"response.image_generation_call.partial_image", "output_index":index,"partial_image_b64":bytes})
}
fn image(bytes: Value) -> Value {
	json!({"type":"image_generation_call","result":bytes})
}
fn done(index: usize, bytes: &str) -> Value {
	json!({"type":"response.output_item.done","output_index":index,"item":image(json!(bytes))})
}
fn terminal(incomplete: bool, output: Vec<Value>) -> Value {
	json!({"type":if incomplete {"response.incomplete"} else {"response.completed"},"response":{"id":"fixture","model":"gpt-5","status":if incomplete {"incomplete"} else {"completed"},"output":output}})
}
fn images(end: &StreamEnd) -> Vec<&str> {
	end.captured_content
		.iter()
		.flat_map(|c| c.parts())
		.filter_map(|p| match p {
			ContentPart::Binary(b) => match &b.source {
				BinarySource::Base64(s) => Some(s.as_ref()),
				_ => None,
			},
			_ => None,
		})
		.collect()
}

#[tokio::test]
async fn final_image_wins_with_or_without_preview_and_terminal() {
	for preview in [false, true] {
		for completed in [false, true] {
			let mut events = vec![];
			if preview {
				events.push(partial(0, "cHJldmlldw=="));
			}
			events.push(done(0, "ZmluYWw="));
			if completed {
				events.push(terminal(false, vec![]));
			}
			let (end, _) = stream(events, true).await;
			assert_eq!(images(&end), ["ZmluYWw="]);
			assert_eq!(end.captured_response_id.as_deref(), completed.then_some("fixture"));
		}
	}
}

#[tokio::test]
async fn clean_eof_and_invalid_terminal_preserve_latest_valid_preview() {
	for invalid in [
		None,
		Some(Value::Null),
		Some(json!("")),
		Some(json!({})),
		Some(json!("not-base64!")),
	] {
		let has_terminal = invalid.is_some();
		let mut events = vec![partial(0, "b2xk"), partial(0, "cHJldmlldw==")];
		if let Some(invalid) = invalid {
			events.push(terminal(true, vec![image(invalid)]));
		}
		let (end, _) = stream(events, true).await;
		assert_eq!(images(&end), ["cHJldmlldw=="]);
		assert_eq!(end.captured_response_id.as_deref(), has_terminal.then_some("fixture"));
	}
}

#[tokio::test]
async fn multiple_images_are_ordered_finalized_once_and_capture_can_be_disabled() {
	let events = vec![
		partial(1, "b2xk"),
		done(1, "ZmluYWw="),
		partial(1, "bGF0ZQ=="),
		partial(0, "cHJldmlldw=="),
		terminal(false, vec![image(json!("bmV3")), image(Value::Null)]),
	];
	let (end, _) = stream(events.clone(), true).await;
	assert_eq!(images(&end), ["bmV3", "ZmluYWw="]);
	let (end, _) = stream(events, false).await;
	assert!(end.captured_content.is_none());
}

#[tokio::test]
async fn terminal_only_image_and_done_only_text_are_captured() {
	let (end, _) = stream(vec![terminal(false, vec![image(json!("ZmluYWw="))])], true).await;
	assert_eq!(images(&end), ["ZmluYWw="]);
	assert_eq!(end.captured_response_id.as_deref(), Some("fixture"));
	let (end, chunks) = stream(
		vec![
			json!({"type":"response.output_text.done","text":"done only"}),
			terminal(false, vec![]),
		],
		true,
	)
	.await;
	assert_eq!(chunks, "done only");
	assert_eq!(end.captured_content.unwrap().first_text(), Some("done only"));
}

#[tokio::test]
async fn cohere_models_are_queried_and_filtered_not_replaced_by_static_names() {
	let body =
		json!({"models":[{"name":"command-fixture"},{"name":"aya-fixture"},{"name":"embed-fixture"}]}).to_string();
	let (endpoint, server) = serve(body, "application/json").await;
	let models = client()
		.all_models(AdapterKind::Cohere, (endpoint, AuthData::from_single("fixture")))
		.await
		.unwrap();
	assert_eq!(
		models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
		["command-fixture", "aya-fixture"]
	);
	let request = server.await.unwrap().to_lowercase();
	assert!(request.starts_with("get /models "));
	assert!(request.contains("authorization: bearer fixture"));
	let (endpoint, server) = serve(json!({"unexpected":[]}).to_string(), "application/json").await;
	assert!(
		client()
			.all_models(AdapterKind::Cohere, (endpoint, AuthData::from_single("fixture")))
			.await
			.is_err()
	);
	server.await.unwrap();
}

#[tokio::test]
async fn read_timeout_builder_limits_a_stalled_response() {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let endpoint = Endpoint::from_owned(format!("http://{}/", listener.local_addr().unwrap()));
	let task = tokio::spawn(async move {
		let (_socket, _) = listener.accept().await.unwrap();
		tokio::time::sleep(Duration::from_secs(10)).await;
	});
	let config = genai::WebConfig::default().with_read_timeout(Duration::from_millis(100));
	let client = Client::builder()
		.with_reqwest(config.apply_to_builder(reqwest::Client::builder().no_proxy()).build().unwrap())
		.build()
		.unwrap();
	let result = tokio::time::timeout(
		Duration::from_secs(3),
		client.all_models(AdapterKind::Anthropic, (endpoint, AuthData::from_single("fixture"))),
	)
	.await;
	task.abort();
	let error = result
		.expect("read timeout should fire before the outer deadline")
		.expect_err("stalled response");
	match error {
		genai::Error::WebAdapterCall {
			webc_error: genai::webc::Error::Reqwest(error),
			..
		} => assert!(error.is_timeout()),
		error => panic!("unexpected error: {error:?}"),
	}
}

#[tokio::test]
async fn text_done_fallback_is_deduplicated_per_output_and_content_index() {
	let events = vec![
		json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"A"}),
		json!({"type":"response.output_text.done","output_index":0,"content_index":0,"text":"A"}),
		json!({"type":"response.output_text.done","output_index":0,"content_index":1,"text":"B"}),
		json!({"type":"response.output_text.done","output_index":0,"content_index":1,"text":"B"}),
		json!({"type":"response.output_text.done","output_index":1,"content_index":0,"text":"C"}),
		terminal(false, vec![]),
	];
	for capture in [true, false] {
		let (end, chunks) = stream(events.clone(), capture).await;
		assert_eq!(chunks, "ABC");
		assert_eq!(
			end.captured_content.as_ref().and_then(|c| c.first_text()),
			capture.then_some("ABC")
		);
	}
}

#[test]
fn reasoning_flags_match_capabilities_without_inventing_support() {
	for (name, expected) in [("gpt-4o", false), ("o3", true), ("unknown-model", false)] {
		let model = Model::from_capabilities(AdapterKind::OpenAI, name);
		assert_eq!(model.supports_reasoning, expected);
		if !model.supports_reasoning {
			assert!(model.supported_reasoning_efforts.is_none());
		}
	}
	let model = Model::new("reasoner", "reasoner").with_reasoning_efforts([]);
	assert!(
		model.supports_reasoning,
		"reasoning without adjustable effort remains expressible"
	);
}

#[tokio::test]
async fn provider_metadata_survives_listing() {
	for (kind, body, path, field, expected) in [
		(
			AdapterKind::Ollama,
			json!({"models":[{"name":"llama3","details":{"family":"llama"},"digest":"sha256:fixture","size":123,"modified_at":"fixture"}]}),
			"/api/tags",
			"/details/family",
			"llama",
		),
		(
			AdapterKind::Anthropic,
			json!({"data":[{"id":"claude-sonnet-4-6","display_name":"Claude Sonnet 4.6","created_at":"fixture"}]}),
			"/models",
			"/display_name",
			"Claude Sonnet 4.6",
		),
	] {
		let (endpoint, server) = serve(body.to_string(), "application/json").await;
		let models = client()
			.all_models(kind, (endpoint, AuthData::from_single("fixture")))
			.await
			.unwrap();
		assert_eq!(
			models[0].additional_properties.as_ref().unwrap().pointer(field).unwrap(),
			expected
		);
		let request = server.await.unwrap();
		assert!(request.starts_with(&format!("GET {path} ")));
		if kind == AdapterKind::Anthropic {
			assert!(request.to_lowercase().contains("x-api-key: fixture"));
		} else {
			let props = models[0].additional_properties.as_ref().unwrap();
			assert_eq!(props["digest"], "sha256:fixture");
			assert_eq!(props["size"], 123);
			assert_eq!(props["modified_at"], "fixture");
		}
	}
}

#[tokio::test]
async fn listing_uses_target_resolver_but_explicit_provider_config_wins() {
	for explicit in [false, true] {
		let (endpoint, server) = serve(
			json!({"data":[{"id":"claude-sonnet-4-6"}]}).to_string(),
			"application/json",
		)
		.await;
		let resolved_endpoint = endpoint.clone();
		let resolver = ServiceTargetResolver::from_resolver_fn(move |mut target: ServiceTarget| {
			assert!(!explicit, "fully explicit configuration must bypass resolver");
			target.endpoint = resolved_endpoint.clone();
			target.auth = AuthData::from_single("resolver-key");
			Ok(target)
		});
		let client = Client::builder()
			.with_reqwest(reqwest::Client::builder().no_proxy().build().unwrap())
			.with_service_target_resolver(resolver)
			.build()
			.unwrap();
		let config = if explicit {
			ProviderConfig::from_endpoint(endpoint).with_auth(AuthData::from_single("explicit-key"))
		} else {
			ProviderConfig::default()
		};
		let models = client.all_models(AdapterKind::Anthropic, config).await.unwrap();
		assert_eq!(models.len(), 1);
		let request = server.await.unwrap().to_lowercase();
		assert!(request.contains(if explicit {
			"x-api-key: explicit-key"
		} else {
			"x-api-key: resolver-key"
		}));
	}
}

#[test]
fn borrowed_tool_helpers_keep_reasoning_and_part_order() {
	let call = ToolCall {
		call_id: "call".into(),
		fn_name: "lookup".into(),
		fn_arguments: json!({}),
		thought_signatures: None,
	};
	let content = MessageContent::from_parts(vec![
		ContentPart::ThoughtSignature("signed".into()),
		ContentPart::ReasoningContent("thinking".into()),
		ContentPart::Text("checking".into()),
		ContentPart::ToolCall(call),
	]);
	let response = ChatResponse {
		content: content.clone(),
		reasoning_content: Some("thinking".into()),
		model_iden: ModelIden::new(AdapterKind::Anthropic, "claude"),
		provider_model_iden: ModelIden::new(AdapterKind::Anthropic, "claude"),
		stop_reason: None,
		usage: Default::default(),
		captured_raw_body: None,
		response_id: None,
	};
	let end = StreamEnd {
		captured_content: Some(content.clone()),
		captured_reasoning_content: Some("thinking".into()),
		..Default::default()
	};
	for message in [
		response.assistant_message_for_tool_use().unwrap(),
		end.assistant_message_for_tool_use().unwrap(),
	] {
		assert_eq!(
			serde_json::to_value(message.content).unwrap(),
			serde_json::to_value(&content).unwrap()
		);
	}
	let request =
		ChatRequest::from_user("hi").append_tool_use_from_chat_response(&response, ToolResponse::new("call", "ok"));
	assert_eq!(request.messages.len(), 3);
	assert_eq!(
		serde_json::to_value(&request.messages[1].content).unwrap(),
		serde_json::to_value(&content).unwrap()
	);
	assert!(StreamEnd::default().assistant_message_for_tool_use().is_none());
}

// Spawn isolated processes to test environment defaults without unsafe env mutation
// or races with concurrently running tests (Rust 2024 forbids safe set_var).
#[test]
fn endpoint_environment_defaults_and_explicit_precedence() {
	for value in [
		None,
		Some("http://127.0.0.1:45678/v1"),
		Some("http://127.0.0.1:45678/v1/"),
		Some(""),
	] {
		let mut command = std::process::Command::new(std::env::current_exe().unwrap());
		command
			.args(["--exact", "endpoint_environment_child", "--nocapture"])
			.env("GENAI_ENDPOINT_TEST_CHILD", "1");
		for name in ["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"] {
			command.env_remove(name);
			if let Some(value) = value {
				command.env(name, value);
			}
		}
		assert!(command.status().unwrap().success());
	}
}

#[tokio::test]
async fn endpoint_environment_child() {
	if std::env::var_os("GENAI_ENDPOINT_TEST_CHILD").is_none() {
		return;
	}
	for (kind, env, fallback) in [
		(AdapterKind::OpenAI, "OPENAI_BASE_URL", "https://api.openai.com/v1/"),
		(
			AdapterKind::Anthropic,
			"ANTHROPIC_BASE_URL",
			"https://api.anthropic.com/v1/",
		),
	] {
		let client = client();
		let target = client.resolve_service_target_without_model(kind).await.unwrap();
		let expected = std::env::var(env)
			.ok()
			.filter(|v| !v.is_empty())
			.map(|v| format!("{}/", v.trim_end_matches('/')))
			.unwrap_or(fallback.into());
		assert_eq!(target.endpoint.base_url(), expected);
		let explicit = ServiceTarget {
			endpoint: Endpoint::from_static("http://127.0.0.1:45679/"),
			auth: AuthData::None,
			model: ModelIden::new(kind, "fixture"),
		};
		assert_eq!(
			client.resolve_service_target(explicit).await.unwrap().endpoint.base_url(),
			"http://127.0.0.1:45679/"
		);
	}
}

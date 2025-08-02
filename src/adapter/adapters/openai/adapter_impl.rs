use crate::adapter::adapters::support::get_api_key;
use crate::adapter::model_capabilities::ModelCapabilities;
use crate::adapter::openai::OpenAIStreamer;
use crate::adapter::{Adapter, AdapterDispatcher, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet, ChatRequest, ChatResponse, ChatResponseFormat, ChatRole, ChatStream, ChatStreamResponse,
	ContentPart, ImageSource, MessageContent, ReasoningEffort, ToolCall, Usage,
};
use crate::common::{Modality, ReasoningEffortType};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Error, Headers, Model, Result};
use crate::{ModelIden, ServiceTarget};
use reqwest::RequestBuilder;
use reqwest_eventsource::EventSource;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
use tracing::error;
use value_ext::JsonValueExt;

pub struct OpenAIAdapter;

// Latest models
const MODELS: &[&str] = &[
	//
	"gpt-4.1",
	"gpt-4.1-mini",
	"gpt-4.1-nano",
	"o4-mini",
	"gpt-4o",
	"gpt-4o-mini",
	"o3-mini",
];

impl OpenAIAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "OPENAI_API_KEY";
	pub const BASE_URL_DEFAULT_ENV_NAME: &str = "OPENAI_BASE_URL";

	/// 将 OpenAI API 返回的模型数据转换为统一的 Model 结构
	fn parse_openai_model_to_model(model_id: String, mut model_data: Value) -> Result<Model> {
		let model_name: crate::ModelName = model_id.clone().into();

		// 从 OpenAI API 获取的基本信息
		let _created: Option<u64> = model_data.x_take("created").ok();
		let _owned_by: Option<String> = model_data.x_take("owned_by").ok();

		// 根据模型名称推断能力和限制
		let mut model = Model::new(model_name, model_id.clone());

		// 根据模型名称设置 token 限制
		let (max_input_tokens, max_output_tokens) = Self::infer_token_limits(&model_id);
		model = model
			.with_max_input_tokens(max_input_tokens)
			.with_max_output_tokens(max_output_tokens);

		// 设置支持的功能
		let supports_streaming = Self::supports_streaming(&model_id);
		let supports_tool_calls = Self::supports_tool_calls(&model_id);
		let supports_json_mode = Self::supports_json_mode(&model_id);
		let supports_reasoning = Self::supports_reasoning(&model_id);

		model = model
			.with_streaming(supports_streaming)
			.with_tool_calls(supports_tool_calls)
			.with_json_mode(supports_json_mode)
			.with_reasoning(supports_reasoning);

		// 设置支持的模态
		let input_modalities = Self::infer_input_modalities(&model_id);
		let output_modalities = Self::infer_output_modalities(&model_id);

		model = model
			.with_input_modalities(input_modalities)
			.with_output_modalities(output_modalities);

		// 设置推理能力等级
		if supports_reasoning {
			let reasoning_efforts = Self::infer_reasoning_efforts(&model_id);
			model = model.with_reasoning_efforts(reasoning_efforts);
		}

		Ok(model)
	}

	/// 根据模型名称推断 token 限制
	fn infer_token_limits(model_id: &str) -> (Option<u32>, Option<u32>) {
		ModelCapabilities::infer_token_limits(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断是否支持流式输出
	fn supports_streaming(model_id: &str) -> bool {
		ModelCapabilities::supports_streaming(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断是否支持工具调用
	fn supports_tool_calls(model_id: &str) -> bool {
		ModelCapabilities::supports_tool_calls(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断是否支持 JSON 模式
	fn supports_json_mode(model_id: &str) -> bool {
		ModelCapabilities::supports_json_mode(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断是否支持推理模式
	fn supports_reasoning(model_id: &str) -> bool {
		ModelCapabilities::supports_reasoning(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断支持的输入模态
	fn infer_input_modalities(model_id: &str) -> HashSet<Modality> {
		ModelCapabilities::infer_input_modalities(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断支持的输出模态  
	fn infer_output_modalities(model_id: &str) -> HashSet<Modality> {
		ModelCapabilities::infer_output_modalities(AdapterKind::OpenAI, model_id)
	}

	/// 根据模型名称推断支持的推理努力等级
	fn infer_reasoning_efforts(model_id: &str) -> Vec<ReasoningEffortType> {
		ModelCapabilities::infer_reasoning_efforts(AdapterKind::OpenAI, model_id)
	}
}

impl Adapter for OpenAIAdapter {
	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://api.openai.com/v1/";
		let mut base_url = std::env::var(Self::BASE_URL_DEFAULT_ENV_NAME).unwrap_or_else(|_| BASE_URL.to_string());

		// 确保 base_url 以 / 结尾
		if !base_url.ends_with('/') {
			base_url.push('/');
		}

		Endpoint::from_owned(base_url)
	}

	/// Note: Currently returns the common models (see above)
	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	async fn all_models(kind: AdapterKind, target: ServiceTarget) -> Result<Vec<Model>> {
		// 使用默认的认证和端点配置
		let auth = target.auth;
		let endpoint = target.endpoint;

		// 构建一个临时的 ModelIden 用于获取服务 URL
		let model_iden = ModelIden::new(kind, "temp");

		// 获取 models API 的 URL
		let url = Self::util_get_service_url(&model_iden, ServiceType::Models, endpoint);

		// 获取 API key
		let api_key = get_api_key(auth, &model_iden)?;

		// 构建请求头
		let headers = vec![("Authorization".to_string(), format!("Bearer {api_key}"))];

		// 创建 WebClient 并发送请求
		let web_client = crate::webc::WebClient::default();
		let mut web_response = web_client
			.do_get(&url, &headers)
			.await
			.map_err(|webc_error| Error::WebAdapterCall {
				adapter_kind: kind,
				webc_error,
			})?;

		// 解析响应
		let mut models: Vec<Model> = Vec::new();

		if let Value::Array(models_data) = web_response.body.x_take("data")? {
			for mut model_data in models_data {
				let model_id: String = model_data.x_take("id")?;

				// 解析模型的基本信息
				let model = Self::parse_openai_model_to_model(model_id, model_data)?;
				models.push(model);
			}
		}

		Ok(models)
	}

	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		Self::util_get_service_url(model, service_type, endpoint)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		chat_options: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		OpenAIAdapter::util_to_web_request_data(target, service_type, chat_req, chat_options)
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;

		let captured_raw_body = options_set.capture_raw_body().unwrap_or_default().then(|| body.clone());

		// -- Capture the provider_model_iden
		let provider_model_name: Option<String> = body.x_remove("model").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// -- Capture the usage
		let usage = body
			.x_take("usage")
			.map(|value| OpenAIAdapter::into_usage(model_iden.adapter_kind, value))
			.unwrap_or_default();

		// -- Capture the content
		let mut content: Vec<MessageContent> = Vec::new();
		let mut reasoning_content: Option<String> = None;

		if let Ok(Some(mut first_choice)) = body.x_take::<Option<Value>>("/choices/0") {
			// Check if reasoning is present
			// Can be in two places:
			// - /message/reasoning
			// - /message/reasoning_content
			// Extracted before content as some model can return reasoning without content
			reasoning_content = first_choice
				.x_take::<Option<String>>("/message/reasoning")
				.ok()
				.unwrap_or_else(|| {
					first_choice
						.x_take::<Option<String>>("/message/reasoning_content")
						.ok()
						.flatten()
				})
				.map(|s| s.trim().to_string());

			// -- Push eventual text message
			if let Ok(Some(mut text_content)) = first_choice.x_take::<Option<String>>("/message/content") {
				text_content = text_content.trim().to_string();
				// If not reasoning_content, but
				if reasoning_content.is_none() && options_set.normalize_reasoning_content().unwrap_or_default() {
					let (content_tmp, reasoning_content_tmp) = extract_think(text_content);
					reasoning_content = reasoning_content_tmp;
					text_content = content_tmp;
				}

				// After extracting reasoning_content, sometimes the content is empty.
				if !text_content.is_empty() {
					content.push(text_content.into());
				}
			}

			// -- Push eventual ToolCalls
			if let Some(tool_calls) = first_choice
				.x_take("/message/tool_calls")
				.ok()
				.map(parse_tool_calls)
				.transpose()?
				.map(MessageContent::from_tool_calls)
			{
				content.push(tool_calls);
			}
		}

		Ok(ChatResponse {
			content,
			reasoning_content,
			model_iden,
			provider_model_iden,
			usage,
			captured_raw_body,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_sets: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let event_source = EventSource::new(reqwest_builder)?;
		let openai_stream = OpenAIStreamer::new(event_source, model_iden.clone(), options_sets);
		let chat_stream = ChatStream::from_inter_stream(openai_stream);

		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}

	fn to_embed_request_data(
		service_target: ServiceTarget,
		embed_req: crate::embed::EmbedRequest,
		options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		super::embed::to_embed_request_data(service_target, embed_req, options_set)
	}

	fn to_embed_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		super::embed::to_embed_response(model_iden, web_response, options_set)
	}
}

/// Support functions for other adapters that share OpenAI APIs
impl OpenAIAdapter {
	pub(in crate::adapter::adapters) fn util_get_service_url(
		_model: &ModelIden,
		service_type: ServiceType,
		// -- utility arguments
		default_endpoint: Endpoint,
	) -> String {
		let base_url = default_endpoint.base_url();
		// Parse into URL and query-params
		let base_url = reqwest::Url::parse(base_url).unwrap();
		let original_query_params = base_url.query().to_owned();

		let suffix = match service_type {
			ServiceType::Chat | ServiceType::ChatStream => "chat/completions",
			ServiceType::Embed => "embeddings",
			ServiceType::Models => "models",
		};
		let mut full_url = base_url.join(suffix).unwrap();
		full_url.set_query(original_query_params);
		full_url.to_string()
	}

	pub(in crate::adapter::adapters) fn util_to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { model, auth, endpoint } = target;
		let (model_name, _) = model.model_name.as_model_name_and_namespace();
		let adapter_kind = model.adapter_kind;

		// -- api_key
		let api_key = get_api_key(auth, &model)?;

		// -- url
		let url = AdapterDispatcher::get_service_url(&model, service_type, endpoint);

		// -- headers
		let mut headers = Headers::from(("Authorization".to_string(), format!("Bearer {api_key}")));

		// -- extra headers
		if let Some(extra_headers) = options_set.extra_headers() {
			headers.merge_with(extra_headers);
		}

		let stream = matches!(service_type, ServiceType::ChatStream);

		// -- compute reasoning_effort and eventual trimmed model_name
		// For now, just for openai AdapterKind
		let (reasoning_effort, model_name): (Option<ReasoningEffort>, &str) =
			if matches!(adapter_kind, AdapterKind::OpenAI) {
				let (reasoning_effort, model_name) = options_set
					.reasoning_effort()
					.cloned()
					.map(|v| (Some(v), model_name))
					.unwrap_or_else(|| ReasoningEffort::from_model_name(model_name));

				(reasoning_effort, model_name)
			} else {
				(None, model_name)
			};

		// -- Build the basic payload

		let OpenAIRequestParts { messages, tools } = Self::into_openai_request_parts(&model, chat_req)?;
		let mut payload = json!({
			"model": model_name,
			"messages": messages,
			"stream": stream
		});

		// -- Set reasoning effort
		if let Some(reasoning_effort) = reasoning_effort {
			if let Some(keyword) = reasoning_effort.as_keyword() {
				payload.x_insert("reasoning_effort", keyword)?;
			}
		}

		// -- Tools
		if let Some(tools) = tools {
			payload.x_insert("/tools", tools)?;
		}

		// -- Add options
		let response_format = if let Some(response_format) = options_set.response_format() {
			match response_format {
				ChatResponseFormat::JsonMode => Some(json!({"type": "json_object"})),
				ChatResponseFormat::JsonSpec(st_json) => {
					// "type": "json_schema", "json_schema": {...}

					let mut schema = st_json.schema.clone();
					schema.x_walk(|parent_map, name| {
						if name == "type" {
							let typ = parent_map.get("type").and_then(|v| v.as_str()).unwrap_or("");
							if typ == "object" {
								parent_map.insert("additionalProperties".to_string(), false.into());
							}
						}
						true
					});

					Some(json!({
						"type": "json_schema",
						"json_schema": {
							"name": st_json.name.clone(),
							"strict": true,
							// TODO: add description
							"schema": schema,
						}
					}))
				}
			}
		} else {
			None
		};

		if let Some(response_format) = response_format {
			payload["response_format"] = response_format;
		}

		// -- Add supported ChatOptions
		if stream & options_set.capture_usage().unwrap_or(false) {
			payload.x_insert("stream_options", json!({"include_usage": true}))?;
		}

		if let Some(temperature) = options_set.temperature() {
			payload.x_insert("temperature", temperature)?;
		}

		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("stop", options_set.stop_sequences())?;
		}

		if let Some(max_tokens) = options_set.max_tokens() {
			payload.x_insert("max_tokens", max_tokens)?;
		}
		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("top_p", top_p)?;
		}
		if let Some(seed) = options_set.seed() {
			payload.x_insert("seed", seed)?;
		}
		Ok(WebRequestData { url, headers, payload })
	}

	/// Note: Needs to be called from super::streamer as well
	pub(super) fn into_usage(adapter: AdapterKind, usage_value: Value) -> Usage {
		// NOTE: here we make sure we do not fail since we do not want to break a response because usage parsing fail
		let usage = serde_json::from_value(usage_value).map_err(|err| {
			error!("Fail to deserilaize uage. Cause: {err}");
			err
		});
		let mut usage: Usage = usage.unwrap_or_default();
		// Will set details to None if no values
		usage.compact_details();

		// Unfortunately, xAI grok-3 does not compute reasoning tokens correctly.
		// Example: completion_tokens: 35, completion_tokens_details.reasoning_tokens: 192
		// BUT completion_tokens should be 35 + 192.
		// TODO: We might want to do this for other token details as well.
		// TODO: We could check if the math adds up first with the total token count, and only change it if it does not.
		//       This will allow us to be forward compatible if/when they fix this bug (yes, it is a bug).
		if matches!(adapter, AdapterKind::Xai) {
			if let Some(reasoning_tokens) = usage.completion_tokens_details.as_ref().and_then(|d| d.reasoning_tokens) {
				let completion_tokens = usage.completion_tokens.unwrap_or(0);
				usage.completion_tokens = Some(completion_tokens + reasoning_tokens)
			}
		}

		usage
	}

	/// Takes the genai ChatMessages and builds the OpenAIChatRequestParts
	/// - `genai::ChatRequest.system`, if present, is added as the first message with role 'system'.
	/// - All messages get added with the corresponding roles (tools are not supported for now)
	fn into_openai_request_parts(_model_iden: &ModelIden, chat_req: ChatRequest) -> Result<OpenAIRequestParts> {
		let mut messages: Vec<Value> = Vec::new();

		// -- Process the system
		if let Some(system_msg) = chat_req.system {
			messages.push(json!({"role": "system", "content": system_msg}));
		}

		// -- Process the messages
		for msg in chat_req.messages {
			// Note: Will handle more types later
			match msg.role {
				// For now, system and tool messages go to the system
				ChatRole::System => {
					if let MessageContent::Text(content) = msg.content {
						messages.push(json!({"role": "system", "content": content}))
					}
					// TODO: Probably need to warn if it is a ToolCalls type of content
				}
				ChatRole::User => {
					let content = match msg.content {
						MessageContent::Text(content) => json!(content),
						MessageContent::Parts(parts) => {
							json!(
								parts
									.iter()
									.map(|part| match part {
										ContentPart::Text(text) => json!({"type": "text", "text": text.clone()}),
										ContentPart::Image { content_type, source } => {
											match source {
												ImageSource::Url(url) => {
													json!({"type": "image_url", "image_url": {"url": url}})
												}
												ImageSource::Base64(content) => {
													let image_url = format!("data:{content_type};base64,{content}");
													json!({"type": "image_url", "image_url": {"url": image_url}})
												}
											}
										}
									})
									.collect::<Vec<Value>>()
							)
						}
						// Use `match` instead of `if let`. This will allow to future-proof this
						// implementation in case some new message content types would appear,
						// this way library would not compile if not all methods are implemented
						// continue would allow to gracefully skip pushing unserializable message
						// TODO: Probably need to warn if it is a ToolCalls type of content
						MessageContent::ToolCalls(_) => continue,
						MessageContent::ToolResponses(_) => continue,
					};
					messages.push(json! ({"role": "user", "content": content}));
				}

				ChatRole::Assistant => match msg.content {
					MessageContent::Text(content) => messages.push(json! ({"role": "assistant", "content": content})),
					MessageContent::ToolCalls(tool_calls) => {
						let tool_calls = tool_calls
							.into_iter()
							.map(|tool_call| {
								json!({
									"type": "function",
									"id": tool_call.call_id,
									"function": {
										"name": tool_call.fn_name,
										"arguments": tool_call.fn_arguments.to_string(),
									}
								})
							})
							.collect::<Vec<Value>>();
						messages.push(json! ({"role": "assistant", "tool_calls": tool_calls, "content": ""}))
					}
					// TODO: Probably need to trace/warn that this will be ignored
					MessageContent::Parts(_) => (),
					MessageContent::ToolResponses(_) => (),
				},

				ChatRole::Tool => {
					if let MessageContent::ToolResponses(tool_responses) = msg.content {
						for tool_response in tool_responses {
							messages.push(json!({
								"role": "tool",
								"content": tool_response.content,
								"tool_call_id": tool_response.call_id,
							}))
						}
					}
					// TODO: Probably need to trace/warn that this will be ignored
				}
			}
		}

		// -- Process the tools
		let tools = chat_req.tools.map(|tools| {
			tools
				.into_iter()
				.map(|tool| {
					// TODO: Need to handle the error correctly
					// TODO: Needs to have a custom serializer (tool should not have to match to a provider)
					// NOTE: Right now, low probability, so, we just return null if cannot convert to value.
					json!({
						"type": "function",
						"function": {
							"name": tool.name,
							"description": tool.description,
							"parameters": tool.schema,
							// TODO: If we need to support `strict: true` we need to add additionalProperties: false into the schema
							//       above (like structured output)
							"strict": false,
						}
					})
				})
				.collect::<Vec<Value>>()
		});

		Ok(OpenAIRequestParts { messages, tools })
	}
}

// region:    --- Support

fn extract_think(content: String) -> (String, Option<String>) {
	let start_tag = "<think>";
	let end_tag = "</think>";

	if let Some(start) = content.find(start_tag) {
		if let Some(end) = content[start + start_tag.len()..].find(end_tag) {
			let start_pos = start;
			let end_pos = start + start_tag.len() + end;

			let think_content = &content[start_pos + start_tag.len()..end_pos];
			let think_content = think_content.trim();

			// Extract parts of the original content without cloning until necessary
			let before_think = &content[..start_pos];
			let after_think = &content[end_pos + end_tag.len()..];

			// Remove a leading newline in `after_think` if it starts with '\n'
			let after_think = after_think.trim_start();

			// Construct the final cleaned content in one allocation
			let cleaned_content = format!("{before_think}{after_think}");

			return (cleaned_content, Some(think_content.to_string()));
		}
	}

	(content, None)
}

struct OpenAIRequestParts {
	messages: Vec<Value>,
	tools: Option<Vec<Value>>,
}

fn parse_tool_calls(raw_tool_calls: Value) -> Result<Vec<ToolCall>> {
	// Some backends (like sglang) return null if no tool calls are present.
	if raw_tool_calls.is_null() {
		return Ok(vec![]);
	}

	let Value::Array(raw_tool_calls) = raw_tool_calls else {
		return Err(Error::InvalidJsonResponseElement {
			info: "tool calls is not an array",
		});
	};

	let tool_calls = raw_tool_calls.into_iter().map(parse_tool_call).collect::<Result<Vec<_>>>()?;

	Ok(tool_calls)
}

fn parse_tool_call(raw_tool_call: Value) -> Result<ToolCall> {
	// Define a helper struct to match the original JSON structure.
	#[derive(Deserialize)]
	struct IterimToolFnCall {
		id: String,
		#[allow(unused)]
		#[serde(rename = "type")]
		r#type: String,
		function: IterimFunction,
	}

	#[derive(Deserialize)]
	struct IterimFunction {
		name: String,
		arguments: Value,
	}

	let iterim = serde_json::from_value::<IterimToolFnCall>(raw_tool_call)?;

	let fn_name = iterim.function.name;

	// For now, support Object only, and parse the eventual string as a json value.
	// Eventually, we might check pricing
	let fn_arguments = match iterim.function.arguments {
		Value::Object(obj) => Value::Object(obj),
		Value::String(txt) => serde_json::from_str(&txt)?,
		_ => {
			return Err(Error::InvalidJsonResponseElement {
				info: "tool call arguments is not an object",
			});
		}
	};

	// Then, map the fields of the helper struct to the flat structure.
	Ok(ToolCall {
		call_id: iterim.id,
		fn_name,
		fn_arguments,
	})
}

// endregion: --- Support

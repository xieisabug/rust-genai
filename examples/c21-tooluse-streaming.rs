use futures::StreamExt;
use genai::Client;
use genai::chat::ChatStreamEvent;
use genai::chat::printer::{PrintChatStreamOptions, print_chat_stream};
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, Tool, ToolResponse};
use serde_json::json;
use tracing_subscriber::EnvFilter;

// const MODEL: &str = "gemini-2.0-flash";
// const MODEL: &str = "deepseek-chat";
const MODEL: &str = "gemini-3-pro-preview";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	tracing_subscriber::fmt()
		.with_env_filter(EnvFilter::new("genai=debug"))
		// .with_max_level(tracing::Level::DEBUG) // To enable all sub-library tracing
		.init();

	let client = Client::default();

	println!("--- Model: {MODEL}");

	// 1. Define a tool for getting weather information
	let weather_tool = Tool::new("get_weather")
		.with_description("Get the current weather for a location")
		.with_schema(json!({
			"type": "object",
			"properties": {
				"city": {
					"type": "string",
					"description": "The city name"
				},
				"country": {
					"type": "string",
					"description": "The country of the city"
				},
				"unit": {
					"type": "string",
					"enum": ["C", "F"],
					"description": "Temperature unit (C for Celsius, F for Fahrenheit)"
				}
			},
			"required": ["city", "country", "unit"]
		}));

	// 2. Create initial chat request with the user query and the tool
	let chat_req = ChatRequest::new(vec![ChatMessage::user("What's the weather like in Shenzhen, China?")])
		.with_tools(vec![weather_tool]);

	// 3. Set options to capture tool calls in the streaming response
	let chat_options = ChatOptions::default()
		.with_capture_tool_calls(true)
		.with_capture_reasoning_content(true);
	let print_options = PrintChatStreamOptions::from_print_events(false);

	// 4. Make the streaming call and handle the events
	let mut chat_stream = client.exec_chat_stream(MODEL, chat_req.clone(), Some(&chat_options)).await?;

	let mut assistant_msg: Option<ChatMessage> = None;

	// print_chat_stream(chat_res, Some(&print_options)).await?;
	println!("--- Streaming response with tool calls");
	while let Some(result) = chat_stream.stream.next().await {
		match result? {
			ChatStreamEvent::Start => {
				println!("Stream started");
			}
			ChatStreamEvent::Chunk(chunk) => {
				print!("{}", chunk.content);
			}
			ChatStreamEvent::ToolCallChunk(chunk) => {
				println!("  ToolCallChunk: {:?}", chunk.tool_call);
			}
			ChatStreamEvent::ReasoningChunk(chunk) => {
				println!("  ReasoningChunk: {:?}", chunk.content);
			}
			ChatStreamEvent::ThoughtSignatureChunk(chunk) => {
				println!("  ThoughtSignatureChunk: {:?}", chunk.content);
			}
			ChatStreamEvent::End(end) => {
				println!("\nStream ended");

				if let Some(captured_assistant_msg) = end.into_assistant_message_for_tool_use() {
					println!("\nCaptured Tool Calls:");
					for tool_call in captured_assistant_msg.content.tool_calls() {
						println!("- Function: {}", tool_call.fn_name);
						println!("  Arguments: {}", tool_call.fn_arguments);
					}
					assistant_msg = Some(captured_assistant_msg);
				}
			}
		}
	}

	// 5. Now demonstrate how to handle the tool call and continue the conversation
	println!("\n--- Demonstrating full tool call workflow");
	let Some(assistant_msg) = assistant_msg else {
		println!("No tool calls captured, cannot continue.");
		return Ok(());
	};
	// Simulate executing the function
	let first_tool_call = assistant_msg
		.content
		.tool_calls()
		.into_iter()
		.next()
		.ok_or("Expected tool calls in assistant message")?;
	let tool_response = ToolResponse::new(
		first_tool_call.call_id.clone(),
		json!({
			"temperature": 22.5,
			"condition": "Sunny",
			"humidity": 65
		})
		.to_string(),
	);

	// Add both the tool calls and response to chat history.
	// This also preserves any reasoning_content captured during the stream.
	let chat_req = chat_req.append_message(assistant_msg).append_message(tool_response);

	// Get final streaming response
	let chat_options = ChatOptions::default();
	let chat_res = client.exec_chat_stream(MODEL, chat_req, Some(&chat_options)).await?;

	print_chat_stream(chat_res, Some(&print_options)).await?;

	Ok(())
}

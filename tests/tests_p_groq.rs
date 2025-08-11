mod support;

use crate::support::{Check, TestResult, common_tests};
use genai::adapter::AdapterKind;
use genai::resolver::AuthData;

// Note: In groq, the llama3.1 or gemma models fail to produce JSON without a proposed schema.
//       With the "tool-use" groq version, it will work correctly.
// Works with: "deepseek-r1-distill-llama-70b" (does not support json mode)
// "mistral-saba-24b" (require term acceptance)
// "llama-3.1-8b-instant", moonshotai/kimi-k2-instruct,
// "meta-llama/llama-4-maverick-17b-128e-instruct" ($0.6) not part of groq fixed name list, needs to be namespaced
const MODEL: &str = "groq::meta-llama/llama-4-maverick-17b-128e-instruct";
const MODEL_NS: &str = "groq::llama-3.1-8b-instant";

// region:    --- Chat

#[tokio::test]
async fn test_chat_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_namespaced_ok() -> TestResult<()> {
	common_tests::common_test_chat_simple_ok(MODEL_NS, None).await
}

#[tokio::test]
async fn test_chat_multi_system_ok() -> TestResult<()> {
	common_tests::common_test_chat_multi_system_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_json_mode_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_json_structured_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_json_structured_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_temperature_ok() -> TestResult<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stop_sequences_ok() -> TestResult<()> {
	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
}

// endregion: --- Chat

// region:    --- Chat Stream Tests

#[tokio::test]
async fn test_chat_stream_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_stream_capture_content_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_capture_content_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stream_capture_all_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_capture_all_ok(MODEL, None).await
}

// endregion: --- Chat Stream Tests

// region:    --- Image Tests

const VISION_MODEL: &str = "llama-3.2-90b-vision-preview"; // Groq's most advanced vision model
const VISION_MODEL_FAST: &str = "llama-3.2-11b-vision-preview"; // Faster vision model

#[tokio::test]
async fn test_chat_image_b64_ok() -> Result<()> {
	common_tests::common_test_chat_image_b64_ok(VISION_MODEL).await
}

#[tokio::test]
async fn test_chat_image_b64_fast_ok() -> Result<()> {
	// Test with the faster vision model
	common_tests::common_test_chat_image_b64_ok(VISION_MODEL_FAST).await
}

// endregion: --- Image Tests

// region:    --- Tool Tests

#[tokio::test]
async fn test_tool_simple_ok() -> Result<()> {
	common_tests::common_test_tool_simple_ok(MODEL, true).await
}

#[tokio::test]
async fn test_tool_full_flow_ok() -> Result<()> {
	common_tests::common_test_tool_full_flow_ok(MODEL, true).await
}

// endregion: --- Tool Tests

// region:    --- Resolver Tests

#[tokio::test]
async fn test_resolver_auth_ok() -> TestResult<()> {
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("GROQ_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
async fn test_list_models() -> TestResult<()> {
	common_tests::common_test_list_models(AdapterKind::Groq, "llama-3.1-70b-versatile").await
}

#[tokio::test]
async fn test_all_models() -> Result<()> {
	common_tests::common_test_all_models(AdapterKind::Groq, "moonshotai/kimi-k2-instruct").await
}

// endregion: --- List

mod support;

use crate::support::{Check, common_tests};
use genai::adapter::AdapterKind;
use genai::resolver::AuthData;

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>; // For tests.

const MODEL: &str = "gpt-4o-mini"; // "gpt-4o-mini", "gpt-4o"
const MODEL_NS: &str = "openai::gpt-4o-mini";

// region:    --- Chat

#[tokio::test]
async fn test_chat_simple_ok() -> Result<()> {
	common_tests::common_test_chat_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_namespaced_ok() -> Result<()> {
	common_tests::common_test_chat_simple_ok(MODEL_NS, None).await
}

#[tokio::test]
async fn test_chat_multi_system_ok() -> Result<()> {
	common_tests::common_test_chat_multi_system_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_json_mode_ok() -> Result<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_json_structured_ok() -> Result<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_temperature_ok() -> Result<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stop_sequences_ok() -> Result<()> {
	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
}

// endregion: --- Chat

// region:    --- Chat Implicit Cache

#[tokio::test]
async fn test_chat_cache_implicit_simple_ok() -> Result<()> {
	common_tests::common_test_chat_cache_implicit_simple_ok(MODEL).await
}

// endregion: --- Chat Implicit Cache

// region:    --- Chat Stream Tests

#[tokio::test]
async fn test_chat_stream_simple_ok() -> Result<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_stream_capture_content_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_content_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stream_capture_all_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_all_ok(MODEL, None).await
}

// endregion: --- Chat Stream Tests

// region:    --- Image Tests

#[tokio::test]
async fn test_chat_image_url_ok() -> Result<()> {
	common_tests::common_test_chat_image_url_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_image_b64_ok() -> Result<()> {
	common_tests::common_test_chat_image_b64_ok(MODEL).await
}

// endregion: --- Image Test

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
async fn test_resolver_auth_ok() -> Result<()> {
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("OPENAI_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
async fn test_list_models() -> Result<()> {
	common_tests::common_test_list_models(AdapterKind::OpenAI, "gpt-4o").await
}

// when run this test, you need to set the OPENAI_API_KEY environment variables
// if you want to change the base url, you need to set the OPENAI_BASE_URL environment variables
// example:
// linux: export OPENAI_BASE_URL="https://xxxx/v1/" && export OPENAI_API_KEY="sk-proj-1234567890" && cargo test --test tests_p_openai::test_all_models
// windows: $env:OPENAI_BASE_URL="https://xxxx/v1/"; $env:OPENAI_API_KEY="sk-proj-1234567890"; cargo test --test tests_p_openai -- test_all_models
#[tokio::test]
async fn test_all_models() -> Result<()> {
	common_tests::common_test_all_models(AdapterKind::OpenAI, "gpt-4o-mini").await
}

// endregion: --- List

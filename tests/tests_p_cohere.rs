mod support;

use crate::support::{Check, TestResult, common_tests};
use genai::adapter::AdapterKind;
use genai::resolver::AuthData;

const MODEL: &str = "command-r"; // Updated to use more capable model
const MODEL_NS: &str = "cohere::command-r";
const VISION_MODEL: &str = "aya-vision-8b"; // For vision tests

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
async fn test_chat_stop_sequences_ok() -> TestResult<()> {
	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_json_mode_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_json_structured_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(Check::USAGE)).await
}

// endregion: --- Chat

// region:    --- Chat Stream Tests

#[tokio::test]
async fn test_chat_stream_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL, None).await
}

// NOTE 2024-06-23 - Occasionally, the last stream message sent by Cohere is malformed and cannot be parsed.
//                   Will investigate further if requested.
// #[tokio::test]
// async fn test_chat_stream_capture_content_ok() -> TestResult<()> {
// 	common_tests::common_test_chat_stream_capture_content_ok(MODEL).await
// }

#[tokio::test]
async fn test_chat_stream_capture_all_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_capture_all_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_temperature_ok() -> TestResult<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

// endregion: --- Chat Stream Tests

// region:    --- Image Tests

#[tokio::test]
async fn test_chat_image_url_ok() -> TestResult<()> {
	common_tests::common_test_chat_image_url_ok(VISION_MODEL).await
}

#[tokio::test]
async fn test_chat_image_b64_ok() -> TestResult<()> {
	common_tests::common_test_chat_image_b64_ok(VISION_MODEL).await
}

// endregion: --- Image Test

// region:    --- Tool Tests

#[tokio::test]
async fn test_tool_simple_ok() -> TestResult<()> {
	common_tests::common_test_tool_simple_ok(MODEL).await
}

#[tokio::test]
async fn test_tool_full_flow_ok() -> TestResult<()> {
	common_tests::common_test_tool_full_flow_ok(MODEL).await
}

// endregion: --- Tool Tests

// region:    --- Resolver Tests

#[tokio::test]
async fn test_resolver_auth_ok() -> TestResult<()> {
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("COHERE_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
async fn test_list_models() -> TestResult<()> {
	common_tests::common_test_list_models(AdapterKind::Cohere, "command-r-plus").await
}

// when run this test, you need to set the COHERE_API_KEY environment variables
// example:
// linux: export COHERE_API_KEY="your-api-key" && cargo test --test tests_p_cohere::test_all_models
// windows: $env:COHERE_API_KEY="your-api-key"; cargo test --test tests_p_cohere -- test_all_models
#[tokio::test]
async fn test_all_models() -> TestResult<()> {
	common_tests::common_test_all_models(AdapterKind::Cohere, "command-r").await
}

// endregion: --- List

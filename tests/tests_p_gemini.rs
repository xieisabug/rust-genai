mod support;

use crate::support::{Check, TestResult, common_tests};
use genai::adapter::AdapterKind;
use genai::resolver::AuthData;

// "gemini-2.5-flash" "gemini-2.5-pro" "gemini-2.5-flash-lite-preview-06-17"
// "gemini-2.5-flash-zero"
const MODEL: &str = "gemini-2.5-flash";
const MODEL_NS: &str = "gemini::gemini-2.5-flash";

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
async fn test_chat_temperature_ok() -> TestResult<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stop_sequences_ok() -> TestResult<()> {
	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
}

// endregion: --- Chat

// region:    --- Chat Implicit Cache

// NOTE: This should eventually work with the 2.5 Flash, but right now, we do not get the cached_tokens
//       So, disabling
// #[tokio::test]
// async fn test_chat_cache_implicit_simple_ok() -> TestResult<()> {
// 	common_tests::common_test_chat_cache_implicit_simple_ok(MODEL).await
// }

// endregion: --- Chat Implicit Cache

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

// region:    --- Binary Tests

const VISION_MODEL: &str = "gemini-2.5-flash"; // Gemini 2.5 models have enhanced vision capabilities
const VISION_PRO_MODEL: &str = "gemini-2.5-pro"; // Most advanced vision model

// NOTE: Gemini does not seem to support URL-based images in the same way as other providers
// #[tokio::test]
// async fn test_chat_binary_image_url_ok() -> TestResult<()> {
// 	common_tests::common_test_chat_image_url_ok(MODEL).await
// }

#[tokio::test]
async fn test_chat_binary_image_b64_ok() -> TestResult<()> {
	common_tests::common_test_chat_image_b64_ok(VISION_MODEL).await
}

#[tokio::test]
async fn test_chat_binary_pdf_b64_ok() -> TestResult<()> {
	common_tests::common_test_chat_pdf_b64_ok(VISION_PRO_MODEL).await
}

#[tokio::test]
async fn test_chat_binary_multi_b64_ok() -> TestResult<()> {
	common_tests::common_test_chat_multi_binary_b64_ok(VISION_PRO_MODEL).await
}

// endregion: --- Binary Tests

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
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("GEMINI_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
async fn test_list_models() -> TestResult<()> {
	common_tests::common_test_list_models(AdapterKind::Gemini, "gemini-2.5-pro").await
}

#[tokio::test]
async fn test_all_models() -> TestResult<()> {
	common_tests::common_test_all_models(AdapterKind::Gemini, "gemini-2.5-flash").await
}

// endregion: --- List

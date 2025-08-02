mod support;

use crate::support::common_tests;
use genai::adapter::AdapterKind;
use genai::resolver::AuthData;
use serial_test::serial;

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>; // For tests.

// Updated model references to use official models from xAI docs
const MODEL: &str = "grok-3-mini";
const MODEL_FOR_STREAMING: &str = "grok-3";
// region:    --- Chat

#[tokio::test]
#[serial(xai)]
async fn test_chat_simple_ok() -> Result<()> {
	common_tests::common_test_chat_simple_ok(MODEL, None).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_multi_system_ok() -> Result<()> {
	common_tests::common_test_chat_multi_system_ok(MODEL).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_json_mode_ok() -> Result<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(crate::support::Check::USAGE)).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_json_structured_ok() -> Result<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(crate::support::Check::USAGE)).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_temperature_ok() -> Result<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

/// NOTE - Disable for now, buggy as of 2024-12-08
///        Will return `the capital of england is **london` somehow
// #[tokio::test]
// async fn test_chat_stop_sequences_ok() -> Result<()> {
// 	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
// }

// endregion: --- Chat

// region:    --- Chat Stream Tests

#[tokio::test]
#[serial(xai)]
async fn test_chat_stream_simple_ok() -> Result<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL_FOR_STREAMING, None).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_stream_capture_content_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_content_ok(MODEL_FOR_STREAMING).await
}

#[tokio::test]
#[serial(xai)]
async fn test_chat_stream_capture_all_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_all_ok(MODEL_FOR_STREAMING, None).await
}

// endregion: --- Chat Stream Tests

// region:    --- Tool Tests

#[tokio::test]
#[serial(xai)]
async fn test_tool_simple_ok() -> Result<()> {
	common_tests::common_test_tool_simple_ok(MODEL, true).await
}

#[tokio::test]
#[serial(xai)]
async fn test_tool_full_flow_ok() -> Result<()> {
	common_tests::common_test_tool_full_flow_ok(MODEL, true).await
}

// endregion: --- Tool Tests

// region:    --- Vision Tests

const VISION_MODEL: &str = "grok-2-vision-1212"; // xAI's vision model

#[tokio::test]
#[serial(xai)]
async fn test_chat_image_b64_ok() -> Result<()> {
	common_tests::common_test_chat_image_b64_ok(VISION_MODEL).await
}

// endregion: --- Vision Tests

// region:    --- Resolver Tests

#[tokio::test]
#[serial(xai)]
async fn test_resolver_auth_ok() -> Result<()> {
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("XAI_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
#[serial(xai)]
async fn test_list_models() -> Result<()> {
	common_tests::common_test_list_models(AdapterKind::Xai, "grok-3").await
}

#[tokio::test]
#[serial(xai)]
async fn test_all_models() -> Result<()> {
	common_tests::common_test_all_models(AdapterKind::Xai, "grok-4-0709").await
}

// endregion: --- List

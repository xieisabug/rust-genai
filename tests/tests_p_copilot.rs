mod support;

use crate::support::{Check, TestResult, common_tests};
use genai::adapter::AdapterKind;

const MODEL: &str = "gpt-4o";
const MODEL_NS: &str = "github_copilot::gpt-4o";

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
async fn test_chat_json_mode_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(Check::USAGE)).await
}

// endregion: --- Chat

// region:    --- Chat Stream

#[tokio::test]
async fn test_chat_stream_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL_NS, None).await
}

#[tokio::test]
async fn test_chat_stream_capture_content_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_capture_content_ok(MODEL_NS).await
}

// endregion: --- Chat Stream

// region:    --- Tool Use

#[tokio::test]
async fn test_chat_tool_calls_ok() -> TestResult<()> {
	common_tests::common_test_tool_simple_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_tool_stream_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_tool_capture_ok(MODEL).await
}

// endregion: --- Tool Use

// region:    --- Vision

#[tokio::test]
async fn test_chat_vision_image_url_ok() -> TestResult<()> {
	common_tests::common_test_chat_image_url_ok(MODEL).await
}

// endregion: --- Vision

// region:    --- List

#[tokio::test]
async fn test_list_models() -> TestResult<()> {
	common_tests::common_test_list_models(AdapterKind::Copilot, "gpt-4o-mini").await
}

#[tokio::test]
async fn test_all_models() -> TestResult<()> {
	common_tests::common_test_all_models(AdapterKind::Copilot, "gpt-4o-mini").await
}

// endregion: --- List

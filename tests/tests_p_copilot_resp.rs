mod support;

use crate::support::{Check, TestResult, common_tests};
use genai::adapter::AdapterKind;

const MODEL: &str = "copilot_resp::gpt-5.4";

// region:    --- Chat

#[tokio::test]
async fn test_chat_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_json_mode_ok() -> TestResult<()> {
	common_tests::common_test_chat_json_mode_ok(MODEL, Some(Check::USAGE)).await
}

// endregion: --- Chat

// region:    --- Chat Stream

#[tokio::test]
async fn test_chat_stream_simple_ok() -> TestResult<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL, None).await
}

// endregion: --- Chat Stream

// region:    --- Tool Use

#[tokio::test]
async fn test_chat_tool_calls_ok() -> TestResult<()> {
	common_tests::common_test_tool_simple_ok(MODEL).await
}

// endregion: --- Tool Use

// region:    --- List

#[tokio::test]
async fn test_list_models() -> TestResult<()> {
	common_tests::common_test_list_models(AdapterKind::CopilotResp, "gpt-5.4").await
}

#[tokio::test]
async fn test_all_models() -> TestResult<()> {
	common_tests::common_test_all_models(AdapterKind::CopilotResp, "gpt-5.4").await
}

// endregion: --- List

mod support;

use crate::support::{Check, TestResult, common_tests};

// "gemini-2.5-flash", "gemini-2.5-pro-preview"
const MODEL: &str = "gemini-2.5-flash"; // can add "-medium" .. suffix

// Updated to use latest stable Gemini 2.5 models with thinking capabilities
const MODEL: &str = "gemini-2.5-flash"; // can add "-zero", "-low", "-medium", "-high" suffix
const REASONING_MODEL: &str = "gemini-2.5-pro"; // Most advanced reasoning model

// NOTE: For now just single test to make sure reasonning token get captured.

#[tokio::test]
async fn test_chat_simple_ok() -> TestResult<()> {
	// NOTE: At this point, gemini 2.5 does not seems to give back reasoning content.
	//       But it should have REASONING_USAGE
	common_tests::common_test_chat_simple_ok(MODEL, Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_pro_ok() -> TestResult<()> {
	// Test with the most advanced reasoning model
	common_tests::common_test_chat_simple_ok(REASONING_MODEL, Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_zero_ok() -> TestResult<()> {
	// Test with zero reasoning budget (fastest)
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-zero", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_low_ok() -> TestResult<()> {
	// Test with low reasoning budget
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-low", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_medium_ok() -> TestResult<()> {
	// Test with medium reasoning budget
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-medium", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_high_ok() -> TestResult<()> {
	// Test with high reasoning budget (most thorough)
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-high", Some(Check::REASONING_USAGE)).await
}

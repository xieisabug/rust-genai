mod support;

use crate::support::{Check, common_tests};

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>; // For tests.

// Updated to use latest stable Gemini 2.5 models with thinking capabilities
const MODEL: &str = "gemini-2.5-flash"; // can add "-zero", "-low", "-medium", "-high" suffix
const REASONING_MODEL: &str = "gemini-2.5-pro"; // Most advanced reasoning model

// NOTE: For now just single test to make sure reasonning token get captured.

#[tokio::test]
async fn test_chat_simple_ok() -> Result<()> {
	// NOTE: Gemini 2.5 includes reasoning tokens in usage but may not expose reasoning content
	common_tests::common_test_chat_simple_ok(MODEL, Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_pro_ok() -> Result<()> {
	// Test with the most advanced reasoning model
	common_tests::common_test_chat_simple_ok(REASONING_MODEL, Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_zero_ok() -> Result<()> {
	// Test with zero reasoning budget (fastest)
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-zero", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_low_ok() -> Result<()> {
	// Test with low reasoning budget
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-low", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_medium_ok() -> Result<()> {
	// Test with medium reasoning budget
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-medium", Some(Check::REASONING_USAGE)).await
}

#[tokio::test]
async fn test_chat_reasoning_budget_high_ok() -> Result<()> {
	// Test with high reasoning budget (most thorough)
	common_tests::common_test_chat_simple_ok("gemini-2.5-flash-high", Some(Check::REASONING_USAGE)).await
}

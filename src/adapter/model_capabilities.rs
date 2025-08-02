use crate::adapter::AdapterKind;
use crate::common::{Modality, ReasoningEffortType};
use std::collections::HashSet;

/// Utilities to derive model capabilities from the model id/name.
///
/// Many providers choose names that are compatible with or inspired by OpenAI. For that reason
/// we keep the OpenAI rule-set as a generic fall-back. Provider specific heuristics can be added
/// incrementally in the match statements.
#[allow(dead_code)]
pub struct ModelCapabilities;

// Provider优先级顺序（可按需调整）
const PROVIDER_PRIORITY: [AdapterKind; 9] = [
	AdapterKind::OpenAI,
	AdapterKind::Anthropic,
	AdapterKind::Cohere,
	AdapterKind::DeepSeek,
	AdapterKind::Gemini,
	AdapterKind::Groq,
	AdapterKind::Xai,
	AdapterKind::Nebius,
	AdapterKind::Ollama,
];

// Helper macro: 按 Provider 回退顺序查找能力
macro_rules! provider_fallback {
	($self_func: path, $adapter_kind: expr, $model_id: expr, $default: expr) => {{
		if let Some(v) = $self_func($adapter_kind, $model_id) {
			return v;
		}
		for kind in PROVIDER_PRIORITY.iter() {
			if *kind == $adapter_kind {
				continue;
			}
			if let Some(v) = $self_func(*kind, $model_id) {
				return v;
			}
		}
		$default
	}};
}

impl ModelCapabilities {
	// ---------- PUBLIC API ----------

	/// Infer the model token limits (max input, max output)
	pub fn infer_token_limits(adapter_kind: AdapterKind, model_id: &str) -> (Option<u32>, Option<u32>) {
		provider_fallback!(
			Self::provider_token_limits,
			adapter_kind,
			model_id,
			Self::openai_infer_token_limits(model_id)
		)
	}

	/// Whether the model supports server-sent streaming responses.
	pub fn supports_streaming(adapter_kind: AdapterKind, model_id: &str) -> bool {
		provider_fallback!(
			Self::provider_supports_streaming,
			adapter_kind,
			model_id,
			Self::openai_supports_streaming(model_id)
		)
	}

	/// Whether the model supports "function/tool" calls.
	pub fn supports_tool_calls(kind: AdapterKind, model_id: &str) -> bool {
		match kind {
			AdapterKind::OpenAI => Self::openai_supports_tool_calls(model_id),
			_ => true,
		}
	}

	/// Whether the model supports JSON mode (structured output).
	pub fn supports_json_mode(adapter_kind: AdapterKind, model_id: &str) -> bool {
		provider_fallback!(
			Self::provider_supports_json_mode,
			adapter_kind,
			model_id,
			Self::openai_supports_json_mode(model_id)
		)
	}

	/// Whether the model supports reasoning effort control.
	pub fn supports_reasoning(adapter_kind: AdapterKind, model_id: &str) -> bool {
		provider_fallback!(
			Self::provider_supports_reasoning,
			adapter_kind,
			model_id,
			Self::openai_supports_reasoning(model_id)
		)
	}

	/// Input modalities supported by the model.
	pub fn infer_input_modalities(adapter_kind: AdapterKind, model_id: &str) -> HashSet<Modality> {
		provider_fallback!(
			Self::provider_input_modalities,
			adapter_kind,
			model_id,
			Self::openai_infer_input_modalities(model_id)
		)
	}

	/// Output modalities supported by the model.
	pub fn infer_output_modalities(adapter_kind: AdapterKind, model_id: &str) -> HashSet<Modality> {
		provider_fallback!(
			Self::provider_output_modalities,
			adapter_kind,
			model_id,
			Self::openai_infer_output_modalities(model_id)
		)
	}

	/// Reasoning effort types supported by the model (if any).
	pub fn infer_reasoning_efforts(adapter_kind: AdapterKind, model_id: &str) -> Vec<ReasoningEffortType> {
		provider_fallback!(
			Self::provider_reasoning_efforts,
			adapter_kind,
			model_id,
			Self::openai_infer_reasoning_efforts(model_id)
		)
	}

	// ---------- PROVIDER CAPABILITY HELPERS (return Option<...>) ----------

	fn provider_supports_streaming(kind: AdapterKind, model_id: &str) -> Option<bool> {
		match kind {
			AdapterKind::OpenAI => Some(Self::openai_supports_streaming(model_id)),
			AdapterKind::Anthropic
			| AdapterKind::Cohere
			| AdapterKind::DeepSeek
			| AdapterKind::Gemini
			| AdapterKind::Groq
			| AdapterKind::Xai
			| AdapterKind::Nebius
			| AdapterKind::Ollama
			| AdapterKind::Zhipu => Some(true),
		}
	}

	fn provider_supports_json_mode(kind: AdapterKind, model_id: &str) -> Option<bool> {
		match kind {
			AdapterKind::OpenAI => Some(Self::openai_supports_json_mode(model_id)),
			AdapterKind::Anthropic | AdapterKind::Cohere | AdapterKind::Gemini => Some(false),
			AdapterKind::DeepSeek
			| AdapterKind::Groq
			| AdapterKind::Xai
			| AdapterKind::Nebius
			| AdapterKind::Ollama
			| AdapterKind::Zhipu => Some(true),
		}
	}

	fn provider_supports_reasoning(kind: AdapterKind, model_id: &str) -> Option<bool> {
		match kind {
			AdapterKind::OpenAI => Some(Self::openai_supports_reasoning(model_id)),
			AdapterKind::DeepSeek => Some(model_id.contains("reasoner")),
			AdapterKind::Gemini => Some(model_id.contains("thinking")),
			_ => None,
		}
	}

	fn provider_input_modalities(kind: AdapterKind, model_id: &str) -> Option<HashSet<Modality>> {
		match kind {
			AdapterKind::Anthropic => {
				let mut set = HashSet::from([Modality::Text]);
				// Claude 3+ 支持图像输入
				if model_id.contains("claude-3") || model_id.contains("claude-2.1") {
					set.insert(Modality::Image);
				}
				Some(set)
			}
			AdapterKind::Gemini => {
				let mut set = HashSet::from([Modality::Text]);
				if !model_id.contains("text") {
					set.insert(Modality::Image);
				}
				Some(set)
			}
			AdapterKind::Groq => {
				let mut set = HashSet::from([Modality::Text]);
				// 一些 Groq 模型支持图像
				if model_id.contains("vision") || model_id.contains("llama-3.2") {
					set.insert(Modality::Image);
				}
				Some(set)
			}
			AdapterKind::OpenAI => Some(Self::openai_infer_input_modalities(model_id)),
			_ => None,
		}
	}

	fn provider_output_modalities(kind: AdapterKind, model_id: &str) -> Option<HashSet<Modality>> {
		match kind {
			AdapterKind::OpenAI => Some(Self::openai_infer_output_modalities(model_id)),
			_ => None,
		}
	}

	fn provider_reasoning_efforts(kind: AdapterKind, model_id: &str) -> Option<Vec<ReasoningEffortType>> {
		match kind {
			AdapterKind::OpenAI => Some(Self::openai_infer_reasoning_efforts(model_id)),
			AdapterKind::Gemini | AdapterKind::DeepSeek => Some(vec![
				ReasoningEffortType::Low,
				ReasoningEffortType::Medium,
				ReasoningEffortType::High,
				ReasoningEffortType::Budget,
			]),
			_ => None,
		}
	}

	// ---------- 内部辅助 ----------

	fn provider_token_limits(kind: AdapterKind, model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		match kind {
			AdapterKind::OpenAI => Self::openai_specific_token_limits(model_id),
			AdapterKind::Anthropic => Self::anthropic_token_limits(model_id),
			AdapterKind::Cohere => Self::cohere_token_limits(model_id),
			AdapterKind::DeepSeek => Self::deepseek_token_limits(model_id),
			AdapterKind::Gemini => Self::gemini_token_limits(model_id),
			AdapterKind::Groq => Self::groq_token_limits(model_id),
			AdapterKind::Xai => Self::xai_token_limits(model_id),
			AdapterKind::Nebius => Self::nebius_token_limits(model_id),
			AdapterKind::Ollama => Self::ollama_token_limits(model_id),
			AdapterKind::Zhipu => Self::zhipu_token_limits(model_id),
		}
	}

	// ---------- PROVIDER SPECIFIC TOKEN LIMITS ----------

	/// OpenAI具体匹配，只在明确识别到 OpenAI 风格模型ID时返回 Some。
	fn openai_specific_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			id if id.starts_with("gpt-4.1") => (Some(128_000), Some(32_768)),
			id if id.starts_with("gpt-4o") => (Some(128_000), Some(16_384)),
			id if id.starts_with("o3") => (Some(200_000), Some(100_000)),
			id if id.starts_with("o4") => (Some(200_000), Some(256_000)),
			id if id.starts_with("o1") => (Some(200_000), Some(100_000)),
			id if id.starts_with("gpt-4") && id.contains("32k") => (Some(32_768), Some(32_768)),
			id if id.starts_with("gpt-4") => (Some(8_192), Some(4_096)),
			id if id.starts_with("gpt-3.5") && id.contains("16k") => (Some(16_384), Some(16_384)),
			id if id.starts_with("gpt-3.5") => (Some(4_096), Some(4_096)),
			id if id.starts_with("chatgpt") => (Some(16_384), Some(16_384)),
			_ => return None,
		};
		Some(res)
	}

	fn anthropic_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			id if id.contains("claude-3-5-sonnet") => (Some(200_000), Some(8_192)),
			id if id.contains("claude-3-5-haiku") => (Some(200_000), Some(8_192)),
			id if id.contains("claude-3-opus") => (Some(200_000), Some(4_096)),
			id if id.contains("claude-3-sonnet") => (Some(200_000), Some(4_096)),
			id if id.contains("claude-3-haiku") => (Some(200_000), Some(4_096)),
			id if id.contains("claude-2.1") => (Some(200_000), Some(4_096)),
			id if id.contains("claude-2.0") => (Some(100_000), Some(4_096)),
			id if id.contains("claude-instant") => (Some(100_000), Some(4_096)),
			_ => return None,
		};
		Some(res)
	}

	fn cohere_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			id if id.contains("command-r-plus") => (Some(128_000), Some(4_096)),
			id if id.contains("command-r") => (Some(128_000), Some(4_096)),
			id if id.contains("command-light") => (Some(4_096), Some(4_096)),
			id if id.contains("command-light-nightly") => (Some(4_096), Some(4_096)),
			id if id.contains("command") => (Some(4_096), Some(4_096)),
			_ => return None,
		};
		Some(res)
	}

	fn deepseek_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			"deepseek-reasoner" => (Some(64_000), Some(8_192)),
			"deepseek-chat" => (Some(32_000), Some(4_096)),
			_ => return None,
		};
		Some(res)
	}

	fn gemini_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			id if id.contains("gemini-2.0") => (Some(1_000_000), Some(32_768)),
			id if id.contains("gemini-1.5-pro") => (Some(2_000_000), Some(8_192)),
			id if id.contains("gemini-1.5-flash") => (Some(1_000_000), Some(8_192)),
			id if id.contains("gemini-1.0-pro") => (Some(30_720), Some(2_048)),
			id if id.contains("gemini-exp") => (Some(2_000_000), Some(8_192)),
			_ => return None,
		};
		Some(res)
	}

	fn groq_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = match model_id {
			id if id.contains("llama-3.1-405b-reasoning") => (Some(32_768), Some(32_768)),
			id if id.contains("llama-3.1-70b") => (Some(131_072), Some(32_768)),
			id if id.contains("llama-3.1-8b") => (Some(131_072), Some(32_768)),
			id if id.contains("llama-3.3-70b") => (Some(131_072), Some(32_768)),
			id if id.contains("llama-3.2") && id.contains("vision") => (Some(8_192), Some(8_192)),
			id if id.contains("llama-3.2") => (Some(131_072), Some(32_768)),
			id if id.contains("mixtral-8x7b-32768") => (Some(32_768), Some(32_768)),
			id if id.contains("gemma") => (Some(8_192), Some(8_192)),
			id if id.contains("llama3-70b-8192") => (Some(8_192), Some(8_192)),
			_ => return None,
		};
		Some(res)
	}

	fn xai_token_limits(model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		let res = if model_id.contains("grok-3") || model_id.contains("grok") {
			(Some(131_072), Some(32_768))
		} else {
			return None;
		};
		Some(res)
	}

	fn nebius_token_limits(_model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		// Nebius does not expose specific per-model limits publicly; use broad defaults.
		Some((Some(128_000), Some(8_192)))
	}

	fn ollama_token_limits(_model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		// Local Ollama models – very rough defaults.
		Some((Some(32_768), Some(8_192)))
	}

	fn zhipu_token_limits(_model_id: &str) -> Option<(Option<u32>, Option<u32>)> {
		// Zhipu does not expose specific per-model limits publicly; use broad defaults.
		Some((Some(128_000), Some(8_192)))
	}

	// ---------- ORIGINAL OPENAI HELPERS (kept public for fallback) ----------
	fn openai_infer_token_limits(model_id: &str) -> (Option<u32>, Option<u32>) {
		match model_id {
			// GPT-4.1 系列
			id if id.starts_with("gpt-4.1") => (Some(128_000), Some(32_768)),
			// GPT-4o 系列
			id if id.starts_with("gpt-4o") => (Some(128_000), Some(16_384)),
			// O3 系列
			id if id.starts_with("o3") => (Some(200_000), Some(100_000)),
			// O4 系列
			id if id.starts_with("o4") => (Some(200_000), Some(256_000)),
			// O1 系列
			id if id.starts_with("o1") => (Some(200_000), Some(100_000)),
			// GPT-4 系列 32k
			id if id.starts_with("gpt-4") && id.contains("32k") => (Some(32_768), Some(32_768)),
			// GPT-4 系列 8k
			id if id.starts_with("gpt-4") => (Some(8_192), Some(4_096)),
			// GPT-3.5 系列 16k
			id if id.starts_with("gpt-3.5") && id.contains("16k") => (Some(16_384), Some(16_384)),
			// GPT-3.5 系列 4k
			id if id.starts_with("gpt-3.5") => (Some(4_096), Some(4_096)),
			// ChatGPT 系列
			id if id.starts_with("chatgpt") => (Some(16_384), Some(16_384)),
			// 默认
			_ => (Some(4_096), Some(4_096)),
		}
	}

	fn openai_supports_streaming(model_id: &str) -> bool {
		!model_id.contains("whisper") && !model_id.contains("tts") && !model_id.contains("dall-e")
	}

	fn openai_supports_tool_calls(model_id: &str) -> bool {
		model_id.starts_with("gpt-4")
			|| model_id.starts_with("gpt-3.5")
			|| model_id.starts_with("o1")
			|| model_id.starts_with("o3")
			|| model_id.starts_with("o4")
			|| model_id.starts_with("chatgpt")
	}

	fn openai_supports_json_mode(model_id: &str) -> bool {
		model_id.starts_with("gpt-4")
			|| model_id.starts_with("gpt-3.5")
			|| model_id.starts_with("o1")
			|| model_id.starts_with("o3")
			|| model_id.starts_with("o4")
			|| model_id.starts_with("chatgpt")
	}

	fn openai_supports_reasoning(model_id: &str) -> bool {
		model_id.starts_with("o1") || model_id.starts_with("o3") || model_id.starts_with("o4")
	}

	fn openai_infer_input_modalities(model_id: &str) -> HashSet<Modality> {
		let mut modalities = HashSet::new();
		modalities.insert(Modality::Text);

		if model_id.contains("vision")
			|| model_id.starts_with("gpt-4o")
			|| model_id.starts_with("gpt-4.1")
			|| model_id.starts_with("o1")
			|| model_id.starts_with("o3")
			|| model_id.starts_with("o4")
		{
			modalities.insert(Modality::Image);
		}

		if model_id.contains("audio") {
			modalities.insert(Modality::Audio);
		}

		modalities
	}

	fn openai_infer_output_modalities(model_id: &str) -> HashSet<Modality> {
		let mut modalities = HashSet::new();
		modalities.insert(Modality::Text);

		if model_id.contains("tts") {
			modalities.insert(Modality::Audio);
		}
		if model_id.contains("dall-e") {
			modalities.insert(Modality::Image);
		}

		modalities
	}

	fn openai_infer_reasoning_efforts(model_id: &str) -> Vec<ReasoningEffortType> {
		if model_id.starts_with("o1") || model_id.starts_with("o3") || model_id.starts_with("o4") {
			vec![
				ReasoningEffortType::Low,
				ReasoningEffortType::Medium,
				ReasoningEffortType::High,
				ReasoningEffortType::Budget,
			]
		} else {
			vec![]
		}
	}
}

//! Example showing how to list detailed model information (capabilities, limits,
//! modalities) with `Client::all_models`.
//!
//! Run with: `cargo run --example c26-model-info`
//! Requires the API key env var of whatever adapters you list.

use genai::{Client, adapter::AdapterKind};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	// Adapters to inspect. Ollama is left out because it needs a local server.
	const ADAPTER_KINDS: &[AdapterKind] = &[
		AdapterKind::OpenAI,
		AdapterKind::Anthropic,
		AdapterKind::Gemini,
		AdapterKind::Cohere,
		AdapterKind::Groq,
		AdapterKind::DeepSeek,
		AdapterKind::Xai,
	];

	let client = Client::new()?;

	for &adapter_kind in ADAPTER_KINDS {
		println!("\n-- {} models --", adapter_kind);

		let models = match client.all_models(adapter_kind, ()).await {
			Ok(models) => models,
			Err(err) => {
				println!("  failed: {err}");
				if let Some(env_name) = adapter_kind.default_key_env_name() {
					println!("  hint: make sure {env_name} is set");
				}
				continue;
			}
		};

		if models.is_empty() {
			println!("  (no models)");
			continue;
		}

		println!("  {} model(s)", models.len());

		for model in &models {
			println!("  * {} (id: {})", model.name, model.id);

			let mut features = Vec::new();
			if model.supports_tool_calls {
				features.push("tool_calls");
			}
			if model.supports_streaming {
				features.push("streaming");
			}
			if model.supports_json_mode {
				features.push("json_mode");
			}
			if model.supports_reasoning {
				features.push("reasoning");
			}
			if model.is_multimodal() {
				features.push("multimodal");
			}
			if !features.is_empty() {
				println!("      supports: {}", features.join(", "));
			}

			if let Some(limit) = model.effective_input_token_limit() {
				println!("      max input tokens: {limit}");
			}
			if let Some(limit) = model.effective_output_token_limit() {
				println!("      max output tokens: {limit}");
			}
			if model.is_multimodal() {
				println!("      input modalities:  {:?}", model.supported_input_modalities);
				println!("      output modalities: {:?}", model.supported_output_modalities);
			}
			if let Some(efforts) = &model.supported_reasoning_efforts
				&& !efforts.is_empty()
			{
				println!("      reasoning efforts: {efforts:?}");
			}
		}
	}

	println!("\nNotes:");
	println!("  - most adapters use a static model list");
	println!("  - Ollama queries the local server instead");
	println!("  - some models may need extra entitlements to be usable");

	Ok(())
}

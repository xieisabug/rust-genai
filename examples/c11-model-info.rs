//! Example showing how to get model information using the client

use genai::{
	Client, ModelIden, ServiceTarget,
	adapter::AdapterKind,
	resolver::{AuthData, Endpoint, ServiceTargetResolver},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	println!("=== 模型信息示例 ===\n");

	// 要查询的适配器类型
	const ADAPTER_KINDS: &[AdapterKind] = &[
		AdapterKind::OpenAI,
		AdapterKind::Anthropic,
		AdapterKind::Gemini,
		AdapterKind::Cohere,
		AdapterKind::Groq,
		AdapterKind::DeepSeek,
		AdapterKind::Xai,
		// AdapterKind::Ollama, // 注释掉 Ollama，因为它需要本地服务运行
	];

	let client = Client::default();

	for &adapter_kind in ADAPTER_KINDS {
		println!("\n🔍 获取 {} 的模型信息...", adapter_kind);

		match client.all_models(adapter_kind).await {
			Ok(models) => {
				if models.is_empty() {
					println!("   ❌ 没有找到可用的模型");
					continue;
				}

				println!("   ✅ 找到 {} 个模型:\n", models.len());

				for (index, model) in models.iter().enumerate() {
					println!("   📋 模型 #{}: {}", index + 1, model.name);
					println!("      模型 ID: {}", model.id);
					println!("      适配器类型: {}", adapter_kind);

					// 显示支持的功能
					let mut features = Vec::new();
					if model.supports_tool_calls {
						features.push("🔧 工具调用");
					}
					if model.supports_streaming {
						features.push("🌊 流式响应");
					}
					if model.supports_json_mode {
						features.push("📄 JSON 模式");
					}
					if model.supports_reasoning {
						features.push("🧠 推理能力");
					}
					if model.is_multimodal() {
						features.push("🖼️ 多模态");
					}

					if !features.is_empty() {
						println!("      支持的功能: {}", features.join(", "));
					}

					// 显示令牌限制
					if let Some(input_limit) = model.effective_input_token_limit() {
						println!("      输入令牌限制: {}", input_limit);
					}
					if let Some(output_limit) = model.effective_output_token_limit() {
						println!("      输出令牌限制: {}", output_limit);
					}

					// 显示支持的模态
					if model.is_multimodal() {
						println!("      输入模态: {:?}", model.supported_input_modalities);
						println!("      输出模态: {:?}", model.supported_output_modalities);
					}

					// 显示推理能力详情
					if model.supports_reasoning {
						if let Some(ref efforts) = model.supported_reasoning_efforts {
							if !efforts.is_empty() {
								println!("      支持的推理强度: {:?}", efforts);
							}
						}
					}

					println!();
				}
			}
			Err(e) => {
				println!("   ❌ 获取模型失败: {}", e);
				// 如果是因为 API 密钥问题，给出友好的提示
				if let Some(env_name) = adapter_kind.default_key_env_name() {
					println!("      💡 提示: 请确保设置了环境变量 {}", env_name);
				}
			}
		}
	}

	println!("\n📝 注意事项:");
	println!("   • 大部分适配器使用静态模型列表");
	println!("   • Ollama 适配器会动态查询本地服务");
	println!("   • 需要设置相应的 API 密钥环境变量");
	println!("   • 某些模型可能需要特殊权限才能访问");

	Ok(())
}

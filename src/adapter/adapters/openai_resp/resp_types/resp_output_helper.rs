use crate::chat::{Binary, ContentPart, ToolCall};
use crate::{Error, Result};
use serde_json::Value;
use value_ext::JsonValueExt;

#[derive(Debug)]
pub struct RespOutputParts {
	pub content: Vec<ContentPart>,
	pub reasoning_content: Option<String>,
	pub thought_signatures: Vec<String>,
}

pub fn parse_resp_output(output: Vec<Value>) -> Result<RespOutputParts> {
	let mut content = Vec::new();
	let mut reasoning_parts = Vec::new();
	let mut thought_signatures = Vec::new();

	for output_item in output {
		if output_item.x_get_str("type").ok() == Some("reasoning") {
			if let Some(reasoning_text) = extract_reasoning_text(&output_item) {
				reasoning_parts.push(reasoning_text);
			}
			if let Ok(encrypted_content) = output_item.x_get_str("encrypted_content") {
				thought_signatures.push(encrypted_content.to_string());
			}
			continue;
		}

		content.extend(ContentPart::from_resp_output_item(output_item)?);
	}

	Ok(RespOutputParts {
		content,
		reasoning_content: (!reasoning_parts.is_empty()).then(|| reasoning_parts.join("\n")),
		thought_signatures,
	})
}

pub fn extract_reasoning_text(item: &Value) -> Option<String> {
	let summary = item.get("summary")?.as_array()?;
	let mut texts = Vec::new();

	for entry in summary {
		if let Some(text) = entry.as_str() {
			texts.push(text.to_string());
			continue;
		}

		if let Some(text) = entry.get("text").and_then(Value::as_str) {
			texts.push(text.to_string());
			continue;
		}

		if let Some(content_parts) = entry.get("content").and_then(Value::as_array) {
			for content_part in content_parts {
				if let Some(text) = content_part.get("text").and_then(Value::as_str) {
					texts.push(text.to_string());
				}
			}
		}
	}

	(!texts.is_empty()).then(|| texts.join("\n"))
}

/// Convert a OpenAI response output Item to a ContentPart
///
/// NOTE: At this point this is infallible, will ignore item that cannot be transformed
impl ContentPart {
	pub fn from_resp_output_item(mut item_value: Value) -> Result<Vec<Self>> {
		let mut parts = Vec::new();
		let Some(item_type) = ItemType::from_item_value(&item_value) else {
			return Ok(parts);
		};

		match item_type {
			ItemType::Message => {
				if let Ok(content) = item_value.x_remove::<Vec<Value>>("content") {
					// each content item {}
					for mut content_item in content {
						match content_item.x_get_str("type") {
							Ok("output_text" | "refusal") => {
								if let Ok(text) = content_item.x_remove::<String>("text")
									&& !text.is_empty()
								{
									parts.push(text.into());
								}
							}
							_ => {}
						}
					}
				}
			}
			ItemType::FunctionCall => {
				let fn_name = item_value.x_remove::<String>("name")?;
				let call_id = item_value.x_remove::<String>("call_id")?;
				let arguments = item_value.x_remove::<String>("arguments")?;
				let fn_arguments: Value =
					serde_json::from_str(&arguments).map_err(|_| Error::InvalidJsonResponseElement {
						info: "tool call arguments is not an object.\nCause",
					})?;

				let tool_call = ToolCall {
					call_id,
					fn_name,
					fn_arguments,
					thought_signatures: None,
				};

				parts.push(tool_call.into());
			}
			ItemType::ImageGenerationCall => {
				if let Ok(result) = item_value.x_remove::<String>("result")
					&& !result.is_empty()
				{
					let content_type = image_content_type(item_value.x_get_str("output_format").ok());
					parts.push(Binary::from_base64(content_type, result, None).into());
				}
			}
		}

		Ok(parts)
	}
}

// region:    --- Support Type

/// The managed
enum ItemType {
	Message,
	FunctionCall,
	ImageGenerationCall,
}

impl ItemType {
	fn from_item_value(item_value: &Value) -> Option<Self> {
		let typ = item_value.x_get_str("type").ok()?;
		match typ {
			"message" => Some(ItemType::Message),
			"function_call" => Some(ItemType::FunctionCall),
			"image_generation_call" => Some(ItemType::ImageGenerationCall),
			_ => None,
		}
	}
}

fn image_content_type(output_format: Option<&str>) -> String {
	match output_format.unwrap_or("png").trim().to_ascii_lowercase().as_str() {
		"jpg" | "jpeg" => "image/jpeg".to_string(),
		"png" => "image/png".to_string(),
		"webp" => "image/webp".to_string(),
		"gif" => "image/gif".to_string(),
		"bmp" => "image/bmp".to_string(),
		"tiff" | "tif" => "image/tiff".to_string(),
		value if value.starts_with("image/") => value.to_string(),
		value => format!("image/{value}"),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_from_resp_output_item_image_generation_call() {
		let item = serde_json::json!({
			"type": "image_generation_call",
			"output_format": "png",
			"result": "aGVsbG8="
		});

		let parts = ContentPart::from_resp_output_item(item).expect("image generation item should parse");
		assert_eq!(parts.len(), 1);
		let binary = parts[0].as_binary().expect("part should be binary");
		assert_eq!(binary.content_type, "image/png");
	}

	#[test]
	fn test_parse_resp_output_extracts_reasoning_summary() {
		let parsed = parse_resp_output(vec![serde_json::json!({
			"type": "reasoning",
			"summary": [
				{"text": "First thought"},
				{"content": [{"text": "Second thought"}]}
			],
			"encrypted_content": "enc_123"
		})])
		.expect("reasoning item should parse");

		assert_eq!(
			parsed.reasoning_content.as_deref(),
			Some("First thought\nSecond thought")
		);
		assert_eq!(parsed.thought_signatures, vec!["enc_123"]);
	}
}

// endregion: --- Support Type

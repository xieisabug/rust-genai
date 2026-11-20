use crate::{ModelIden, adapter::AdapterKind, chat::ContentPart};
use serde_json::{Value, json};

/// Keep the provider-signed calls intact. Only server_tool_calls are echoed on the wire.
#[derive(Default)]
pub(super) struct DeepSeekWeb {
	events: Vec<Value>,
	calls: Option<Value>,
	annotations: Option<Value>,
}

impl DeepSeekWeb {
	pub fn capture(&mut self, choice: &Value, body: &str) {
		if let Some(event) = choice.get("server_tool_event").filter(|v| v.is_object()) {
			self.events.push(event.clone());
		}
		let body = &choice[body];
		for (key, target) in [("server_tool_calls", &mut self.calls), ("annotations", &mut self.annotations)] {
			if let Some(value) = body.get(key).filter(|v| v.as_array().is_some_and(|a| !a.is_empty())) {
				*target = Some(value.clone());
			}
		}
	}

	pub fn into_part(self, model: &ModelIden) -> Option<ContentPart> {
		if model.adapter_kind != AdapterKind::DeepSeek || (self.calls.is_none() && self.annotations.is_none() && self.events.is_empty()) {
			return None;
		}
		let mut data = json!({"type":"deepseek_web", "events":self.events});
		if let Some(calls) = self.calls { data["server_tool_calls"] = calls; }
		if let Some(annotations) = self.annotations { data["annotations"] = annotations; }
		Some(ContentPart::from_custom(data, Some(model.clone())))
	}
}

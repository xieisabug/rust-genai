use crate::{ModelIden, adapter::AdapterKind, chat::{ContentPart, CustomPart}};
use serde_json::{Value, json};

/// Keep the provider-signed calls intact. Only server_tool_calls are echoed on the wire.
#[derive(Default)]
pub(super) struct DeepSeekWeb {
	events: Vec<Value>,
	calls: Option<Value>,
	annotations: Option<Value>,
}

impl DeepSeekWeb {
	/// Records one choice and returns its `server_tool_event`, if any, for live delivery.
	pub fn capture(&mut self, choice: &Value, body: &str) -> Option<Value> {
		let event = choice.get("server_tool_event").filter(|v| v.is_object()).cloned();
		if let Some(event) = &event {
			self.events.push(event.clone());
		}
		let body = &choice[body];
		for (key, target) in [("server_tool_calls", &mut self.calls), ("annotations", &mut self.annotations)] {
			if let Some(value) = body.get(key).filter(|v| v.as_array().is_some_and(|a| !a.is_empty())) {
				*target = Some(value.clone());
			}
		}
		event
	}

	/// A live `server_tool_event` as a stream event part: `{"type":"deepseek_web_event","event":…}`.
	pub fn live_event(model: &ModelIden, event: Value) -> CustomPart {
		CustomPart { model_iden: Some(model.clone()), data: json!({"type": "deepseek_web_event", "event": event}) }
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

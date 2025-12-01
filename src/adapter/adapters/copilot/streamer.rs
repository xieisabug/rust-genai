//! Stream handling for GitHub Copilot Chat API

use super::types::CopilotStreamResponse;
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::ToolCall;
use crate::{Error, ModelIden, Result};
use futures::stream::Stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde_json::from_str;

pub struct CopilotStreamer {
	event_source: EventSource,
	_model_iden: ModelIden,
}

impl CopilotStreamer {
	pub fn new(event_source: EventSource, model_iden: ModelIden) -> Self {
		Self {
			event_source,
			_model_iden: model_iden,
		}
	}
}

impl Stream for CopilotStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(
		mut self: std::pin::Pin<&mut Self>,
		cx: &mut std::task::Context<'_>,
	) -> std::task::Poll<Option<Self::Item>> {
		use std::task::Poll;

		match self.event_source.poll_next_unpin(cx) {
			Poll::Ready(Some(event_result)) => {
				let event = Self::process_event(event_result)?;
				Poll::Ready(Some(Ok(event)))
			}
			Poll::Ready(None) => Poll::Ready(None),
			Poll::Pending => Poll::Pending,
		}
	}
}

impl CopilotStreamer {
    fn process_event(event_result: std::result::Result<Event, reqwest_eventsource::Error>) -> Result<InterStreamEvent> {
		match event_result {
			Ok(Event::Open) => Ok(InterStreamEvent::Start),
			Ok(Event::Message(message)) => {
				let data = message.data;

				// Check for [DONE] marker
				if data.trim() == "[DONE]" {
					return Ok(InterStreamEvent::End(InterStreamEnd::default()));
				}

				// Parse the stream response
				let stream_response: CopilotStreamResponse = from_str(&data).map_err(|e| {
					Error::Internal(format!(
						"Failed to parse Copilot stream response: {} - Data: {}",
						e, data
					))
				})?;

				// Get the first choice's delta
				if let Some(choice) = stream_response.choices.first() {
					let delta = &choice.delta;

					// Handle content delta
					if let Some(content) = &delta.content {
						if !content.is_empty() {
							return Ok(InterStreamEvent::Chunk(content.clone()));
						}
					}

					// Handle tool calls delta
					if let Some(tool_calls) = &delta.tool_calls {
						for delta_tool_call in tool_calls {
							if let Some(function) = &delta_tool_call.function {
								if let (Some(name), Some(id)) = (&function.name, &delta_tool_call.id) {
									let fn_arguments = function
										.arguments
										.as_ref()
										.and_then(|args| serde_json::from_str(args).ok())
										.unwrap_or_default();

									return Ok(InterStreamEvent::ToolCallChunk(ToolCall {
										call_id: id.clone(),
										fn_name: name.clone(),
										fn_arguments,
									}));
								}
							}
						}
					}

					// If finish_reason is present, send end event
					if choice.finish_reason.is_some() {
						return Ok(InterStreamEvent::End(InterStreamEnd::default()));
					}
				}

				// Empty delta, send an empty chunk
				Ok(InterStreamEvent::Chunk(String::new()))
			}
			Err(e) => Err(Error::Internal(format!("Stream error: {}", e))),
		}
	}
}

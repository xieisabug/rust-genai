# Magpie gateway

`genai::providers::magpie` connects to Magpie using existing OpenAI Chat
Completions, OpenAI Responses, Anthropic Messages and Gemini adapters. The
gateway owns any upstream protocol translation. A chosen protocol is never
silently replaced after a failure.

```rust,no_run
use genai::providers::magpie::Gateway;
use genai::WebConfig;
use genai::chat::{ChatMessage, ChatRequest};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
// Pass a gateway key for remote gateways, or an application identifier locally.
// None discovers XDG_CONFIG_HOME/magpie/settings.json or ~/.config/magpie/settings.json;
// the default port is 3425. Discovery reads a local file, so construct the
// Gateway before entering an async worker, or use tokio::task::spawn_blocking.
let gateway = Gateway::new(None, "magpie-myapp")?;
let config = WebConfig::default().with_timeout(std::time::Duration::from_secs(30));
let models = gateway.catalog(config.clone(), "MyApp/1.0").await?;
let model = &models[0];
// Native protocols appear first. Persist any user selection in the application.
let client = gateway.client(Some(&model.available_modes[0]), config,
    "MyApp/1.0", Some("persistent-conversation-id"))?;
let response = client.exec_chat(&model.id,
    ChatRequest::new(vec![ChatMessage::user("Hello")]), None).await?;
# Ok(())
# }
```

Catalog requests validate `/api/hello` and then load `/v1/models`. Non-chat
models are excluded. The catalog preserves gateway model IDs, distinguishes
routing groups and provider labels, and reports input media capabilities.
Endpoints may end in `/v1` or `/v1beta`; any reverse-proxy prefix is retained.
Loopback connections bypass explicit, environment and system proxies.
Invalid protocols and invalid identity headers fail before HTTP transport.

Application identity, configuration UI, stored model selections, persistence
and session-ID generation remain the caller's responsibility. The module has
no Tauri or database dependency.

## Explicit Responses cache TTL

`ChatOptions::with_prompt_cache_ttl("30m")` or `"12h"` sends
`prompt_cache_options.ttl` through the Responses adapter. Explicit TTL
overrides automatic TTL and legacy `prompt_cache_retention`, preserving an
upstream `prompt_cache_options.mode` when present. Per-call options override
client defaults. Unsupported TTL values fail before transport. Without an
explicit TTL, existing adapter behavior is unchanged. Application policy
deciding which models receive this option remains outside genai.

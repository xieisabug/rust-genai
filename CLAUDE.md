# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`genai` is a multi-AI provider library for Rust that provides a unified interface to major AI services including OpenAI, Anthropic, Gemini, xAI, Ollama, Groq, DeepSeek, and Cohere. The library focuses on chat completion APIs with support for streaming, tool usage, embeddings, and image analysis.

## Common Development Commands

### Building and Testing
- `cargo build` - Build the project
- `cargo build --release` - Build with optimizations
- `cargo test` - Run all tests
- `cargo test <test_name>` - Run specific test (e.g., `cargo test tests_p_openai`)
- `cargo test --test <test_file>` - Run specific test file (e.g., `cargo test --test tests_p_anthropic`)
- `cargo check` - Fast syntax and type checking without compilation
- `cargo fmt` - Format code using rustfmt (uses custom config in rustfmt.toml)
- `cargo clippy` - Run linter

### Running Examples
- `cargo run --example c00-readme` - Run the main example demonstrating multi-provider usage
- `cargo run --example c01-conv` - Run conversation flow example
- `cargo run --example c07-image` - Run image analysis example
- Examples require appropriate API keys as environment variables (see examples for details)

### Documentation
- `cargo doc` - Generate documentation
- `cargo doc --open` - Generate and open documentation in browser

## Architecture Overview

The library is organized into several core modules:

### Adapter System (`src/adapter/`)
- **Static dispatch pattern** with `Adapter` trait and `AdapterDispatcher`
- Each AI provider has its own adapter implementation under `adapters/`
- Adapters handle provider-specific request/response transformations
- Model capabilities system defines what features each provider supports
- All adapter methods are stateless (no `&self`) to reduce state management

### Client Layer (`src/client/`)
- Main entry point through `Client` struct
- Builder pattern for client configuration
- Service target resolution for mapping models to endpoints
- Web configuration and HTTP client management

### Resolver System (`src/resolver/`)
- Customization hooks for library behavior
- `AuthResolver` - Provides authentication data (API keys)
- `ModelMapper` - Maps model names to adapter kinds
- `ServiceTargetResolver` - Custom endpoint/auth/model resolution
- All resolvers are user-configurable for custom behavior

### Chat System (`src/chat/`)
- Core chat functionality with streaming support
- Message content types (text, images, tool calls)
- Tool usage support with structured responses
- Usage tracking and response formatting

### Embedding (`src/embed/`)
- Text embedding functionality for supported providers
- Batch embedding support

## Testing Strategy

### Test Organization
- Provider-specific tests: `tests_p_<provider>.rs` (e.g., `tests_p_openai.rs`)
- Reasoning tests: `tests_p_<provider>_reasoning.rs` for providers supporting reasoning
- Embedding tests: `tests_p_<provider>_embeddings.rs`
- Test support utilities in `tests/support/`

### Running Provider Tests
Provider tests require API keys as environment variables:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`
- Groq: `GROQ_API_KEY`
- XAI: `XAI_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`
- Cohere: `COHERE_API_KEY`
- Ollama: No API key needed (runs locally)

## Model Name Mapping

The library automatically maps model names to adapter kinds:
- `gpt*` → OpenAI
- `claude*` → Anthropic
- `command*` → Cohere
- `gemini*` → Gemini
- Groq model list → Groq
- Everything else → Ollama

Custom mapping can be provided via `ModelMapper` resolver.

## Key Conventions

### Code Style
- Uses hard tabs (configured in rustfmt.toml)
- Max width: 120 characters
- Edition 2024
- Unsafe code is forbidden at crate level

### Error Handling
- Custom error types in `src/error.rs`
- Result type alias for convenience
- Comprehensive error context for debugging

### API Design Principles
- Ergonomic and consistent APIs across providers
- Native implementation (no per-service SDKs)
- Stateless adapter methods with explicit parameters
- Builder patterns for complex configurations

### Version Strategy
- Currently in alpha (0.4.0-alpha.8-WIP)
- Breaking changes expected until 1.0
- Check CHANGELOG.md for recent changes
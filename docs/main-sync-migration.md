# main-sync 定制功能迁移记录

本次以 `20f2ae8`（上游基线 `47c90bd`）为起点，对照旧 fork `04359b1`，继续完善新上游上的定制层。保留旧分支作为行为参考，不把旧适配器整段拷回。

## 行为清单

| 旧定制能力 | 当前处理 | 验证位置 |
| --- | --- | --- |
| Models API、Model、模态和能力推断 | 保留默认实现；非推理模型不再被空 efforts 集合强制标为支持推理 | `tests_main_sync_regressions::reasoning_flags_match_capabilities_without_inventing_support` |
| Ollama 元数据 | 恢复 `details`、`digest`、`size`、`modified_at` 等原始字段，名称单独作为 id | `provider_metadata_survives_listing` |
| Anthropic 元数据 | 恢复 `display_name`，同时保留其余响应字段；不改变 model id | 同上 |
| Cohere 动态模型查询 | 恢复配置地址下的 `GET models`、Bearer 认证和旧聊天模型名称过滤 | `cohere_models_are_queried_and_filtered_not_replaced_by_static_names` |
| OpenAI 兼容 provider 的 Models API | 复用上游模型名称查询和定制能力推断，不恢复重复 HTTP 实现 | `all_adapters.rs`、`adapter_types.rs` |
| Gemini 模型名称规范化 | 沿用上游去除 `models/` 前缀的实现 | `gemini/adapter_impl.rs` |
| Copilot / CopilotResp、请求头、模型能力 | 保留现有移植，编解码委托 OpenAI / Responses；不恢复独立旧 streamer | 两个适配器的库内测试；真实账户待验收 |
| Responses 结构化图片 | 保留非流式转换；流式有效终态图片优先于有效预览，按 output_index 排序、去重；EOF 也收尾 | 新增图片回归测试 |
| done-only 文本代理兼容 | 恢复 `output_text.done` 兜底，按 output_index + content_index 去重；有非空 delta 时不重复发完整文本 | 新增文本回归测试 |
| `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | 恢复默认地址回退，补末尾斜杠；空值回退官方默认值 | 隔离子进程环境变量测试 |
| 读超时 | 恢复 `WebConfig::with_read_timeout`，保留上游底层读超时及流首包/读超时实现 | 新增停滞响应测试、`tests_stream_timeouts` |
| 工具回传助手 | 恢复两个借用版本助手和非流式 append 助手；保留上游签名、推理与内容顺序 | `borrowed_tool_helpers_keep_reasoning_and_part_order` |
| 无模型 target 解析 | 恢复公开入口，复用上游 auth + target resolver 链，保留 resolver 返回的 target | 环境变量测试、模型列表 resolver 测试 |
| UTF-8 跨块缓冲、非法/截断字节 | 上游已提供新的实现及测试，不恢复旧解码器 | `web_stream_tests.rs` |
| 工具索引有间隙时的参数合并 | 上游已有索引存储和回归覆盖，不恢复旧实现 | `test_tool_call_slots_keep_explicit_indexes`、`openrouter_stream_keeps_tool_calls_with_sparse_indexes_apart` |
| 捕获 usage、reasoning、工具与续接信息 | 沿用上游 streamer；结构化图片通过统一 StreamEnd 传递 | 库内测试及现有本地回放 |

这是一份行为迁移清单，不是旧提交逐行重放。旧示例、临时文件 `temp_openai.rs`、独立 Copilot 流解析器不恢复。

## 明确保留的接口变化

`all_models` 与上游 `all_model_names` 保持相同的 ProviderConfig 风格。Rust 不支持同名不同参数个数的方法，因此旧调用需增加第二个参数：

```rust,ignore
// 沿用 Client 已配置的 AuthResolver / ServiceTargetResolver。
let models = client.all_models(adapter_kind, ()).await?;

// 显式提供两个值，跳过 resolver，不读取默认地址/认证。
let models = client.all_models(adapter_kind, (endpoint, auth)).await?;
```

纠正早期评估：模型列表并没有绕过 ServiceTargetResolver。`resolve_provider_config` 在参数不完整时调用 `resolve_adapter_config`，后者执行 auth 和 target resolver。只提供 endpoint 或 auth 时，缺失项来自 resolver/default，显式项覆盖对应默认值；两个都提供时直接使用显式值。新增本地 HTTP 测试验证默认 resolver 路径及完全显式配置的优先级。

旧 `resolve_service_target_without_model` 现已恢复；它使用空 model name，不执行模型推断或 model mapper。

恢复的借用助手不会消费响应对象；原有 `into_assistant_message_for_tool_use` 继续保留。不要把旧调用机械改成消费版本而引入所有权问题。

## 地址与错误处理约定

- OpenAI Chat 与 Anthropic 的环境变量仅作用于默认 endpoint。显式 target、provider 配置或 resolver 返回的 endpoint 优先。
- OpenAI Responses 的环境变量行为保持旧 fork 和上游原状；需要自定义地址时显式配置 endpoint。Copilot 有自己的 endpoint。
- provider 模型查询的 HTTP/解析错误向调用方返回，不恢复旧 Gemini 等路径“查询失败却返回硬编码成功列表”的行为。Cohere `all_model_names` 仍是上游静态便捷列表，`all_models` 是动态查询。
- `additional_properties` 保存 Anthropic/Ollama 列表条目去除 id/name 后的原始字段。缺失的字段仍然缺失，不伪造数据。
- `ReasoningEffortType` 仍是粗粒度 UI 分类，会合并 Zero/Minimal、XHigh/Max；不能用于无损保存或往返具体请求配置。无损配置应使用上游 `ReasoningEffort`。
- 干净 EOF 可返回已收到图片，但没有终态 response id/stop reason；不能将其当作服务端完整成功。网络错误继续返回错误，不转换成成功 End。
- 图片校验保证非空、可解析的二进制引用和有效 base64 编码，不验证图片像素格式或视觉完整性。

## AIPP 升级交接

已只读核对当前 AIPP 源码：`llm_api.rs` 两处 `all_models(adapter_kind)` 需要改成 `all_models(adapter_kind, ())`；`with_read_timeout` 已在本库恢复。现有 ServiceTargetResolver 可继续提供地址与认证，不需要为了列表接口另建一套配置。

本次不更换 AIPP 的 Git 依赖 revision，不宣称 AIPP 已完成升级。实际升级时应在 AIPP 仓库更新依赖与调用，执行其编译检查和功能验证；可能还存在上游 0.7 的其他 API 变化，以上两处并非完整编译错误清单。

合入应用前验收：自定义网关聊天与模型列表、普通流式聊天、工具回传（含 reasoning/signature）、最终图片与中断预览、最小 Copilot/Responses/Anthropic 真实请求。不要将本地 fixture 通过等同于真实服务可用。

## 验证

新增回归入口：`cargo test --test tests_main_sync_regressions`。所有请求使用本地地址和假认证；环境变量在独立子进程中设置，不修改测试主进程的全局环境。

现有库测试和本地回放一起执行，排除 `tests_p_*` 在线用例及 `tests_yakbak_record` 录制工具。TLS 特性互斥，分别检查，不能使用 `--all-features`。

2026-11-06 本地验证（Cargo 1.94.0，Windows）：

| 检查 | 结果 |
| --- | --- |
| 默认特性库内测试 | 465 通过 |
| 22 个本地集成/回放 target | 110 通过，含新增 13 个回归测试；环境变量另以 4 次隔离子进程验证 |
| `cargo test --lib --features otel,bedrock-sigv4` | 507 通过，包含默认库测试 |
| `cargo check --all-targets --features otel` | 通过 |
| `cargo check --lib --no-default-features` | 通过 |
| `cargo check --lib --no-default-features --features native-tls` | 通过 |
| Clippy：lib、bins、examples、新增回归 target，`-D warnings` | 通过 |
| 修改文件 rustfmt、`git diff --check` | 通过 |
| 全仓 `cargo fmt --all -- --check` | 上游原有 `openai_model.rs:218` 一处差异；该文件未修改 |
| `cargo test --doc` | 3 通过、3 失败、1 忽略；失败为 `tool_base.rs`、`web_client.rs`、`zai/mod.rs` 的原有示例编译问题，三个源文件与上游一致 |

未执行真实 provider 请求、AIPP 全量编译/设备验收、GitHub runner 上的 CI。以上保留问题不能描述为“全部检查通过”。

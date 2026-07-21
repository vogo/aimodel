# aimodel 文档

`aimodel` 是多协议(OpenAI 兼容、Anthropic)AI 模型 API 的 Go SDK —— 零外部依赖的**薄 API 封装**:只做请求翻译、连接管理与响应规一化,不含重试、限流、校验、缓存、日志/指标。

## 目录

| 文档 | 内容 |
|---|---|
| [api.md](./api.md) | **核心 API 设计** —— 设计定位、规范表示、客户端与协议分派、数据模型、流式抽象、错误模型、多模型编排 |
| [anthropic/anthropic-chat-api.md](./anthropic/anthropic-chat-api.md) | Anthropic Messages API 封装的设计与实现:双向翻译、prompt 缓存、SSE |
| [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) | Anthropic API 各版本变更与 wrapper 跟进记录 |
| [openai/openai-chat-api.md](./openai/openai-chat-api.md) | OpenAI Chat Completions 封装的设计与实现:零翻译路径、字段对齐、SSE |
| [openai/openai-api-changes.md](./openai/openai-api-changes.md) | OpenAI API 各版本变更与 wrapper 跟进记录 |

## 仓库根文档

- [../README.md](../README.md) —— 完整用法:请求/响应字段、token 限制、reasoning effort、多模态、流式、Anthropic 协议、多模型 compose。
- [../CLAUDE.md](../CLAUDE.md) —— 架构、协议分派与 Anthropic 翻译细节(面向 AI 助手)。
- [../CHANGES.md](../CHANGES.md) —— 与官方 API 的同步状态总账(两个协议合并倒序)。

## 官方 API 参考

| 协议 | 官方文档 | wrapper 代码 |
|---|---|---|
| OpenAI(OpenAI 兼容) | https://platform.openai.com/docs/api-reference/chat | `openai_chat.go` / `openai_stream.go` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `anthropic.go` / `anthropic_chat.go` / `anthropic_stream.go` |

## 维护约定

官方 API 变更时,**同步更新四处**并保持一致:

1. wrapper 代码;
2. 仓库根 `README.md` / `CLAUDE.md`;
3. 仓库根 `CHANGES.md`;
4. `doc/` 下对应协议的实现文档与变更记录。

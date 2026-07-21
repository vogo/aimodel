# aimodel API 设计核心文档

`github.com/vogo/aimodel` —— 多协议(OpenAI 兼容、Anthropic)AI 模型 API 的统一 Go SDK,零外部依赖。

本文档描述 SDK 的**核心 API 设计**:类型体系、协议分派、流式抽象、错误模型与多模型编排。
协议侧的字段级映射细节见:

- [anthropic/anthropic-chat-api.md](./anthropic/anthropic-chat-api.md)
- [openai/openai-chat-api.md](./openai/openai-chat-api.md)

---

## 1. 设计定位

aimodel 是一层**薄 API 封装**(thin API wrapper),职责边界严格限定为三件事:

1. **请求翻译** —— 把统一的请求结构翻译成各协议的线上格式;
2. **连接管理** —— HTTP 客户端、超时、鉴权头、SSE 读取;
3. **响应归一化** —— 把各协议的响应/流事件规约回统一结构。

它**刻意不包含**:重试、限流、请求参数校验、缓存/持久化、日志/指标。这些属于调用方或上层框架的职责,放进 SDK 会带来隐式行为与不可控开销。

**推论(设计约束)**:

- 参数不做合法性校验,未知取值原样透传(如 `ReasoningEffort` / `Verbosity` / `ServiceTier` 保持 `string` 而非枚举类型),以兼容各家 OpenAI 兼容后端的私有扩展。
- 请求结构不携带副作用状态,`ChatRequest` 可安全复用。
- 单次调用 = 单次 HTTP 请求(多模型编排的失败转移除外,见 §7)。

## 2. 规范表示:以 OpenAI 格式为中心

SDK 选择 **OpenAI Chat Completions 格式作为规范(canonical)表示**:

```
                     ┌─────────────────────────┐
   ChatRequest ─────▶│  Protocol = openai      │──▶ POST {baseURL}/chat/completions
  (OpenAI 形状)      │  (直接序列化,零翻译)    │
        │            └─────────────────────────┘
        │            ┌─────────────────────────┐
        └───────────▶│  Protocol = anthropic   │──▶ POST {baseURL}/v1/messages
                     │  toAnthropicRequest()   │
                     └─────────────────────────┘
                                  │
   ChatResponse ◀── fromAnthropicResponse() ◀───┘
  (OpenAI 形状)
```

这么选的理由:OpenAI 格式是事实标准,绝大多数后端(DeepSeek、Kimi、GLM、Qwen、Doubao、MiniMax…)原生兼容,选它做规范表示可使 OpenAI 路径**零翻译开销**,只有 Anthropic 路径需要双向翻译。

**协议专属字段的处理规则**:

| 情形 | 处理方式 | 例子 |
|---|---|---|
| 两边都有 | 规范字段 + 双向映射 | `TopP`、`Stop`↔`stop_sequences` |
| 仅 Anthropic 有,且不属于线上 body | 规范结构上加 `json:"-"` 的**结构体局部字段**,仅 Anthropic 翻译器读取 | `Message.CacheBreakpoint`、`Tool.CacheBreakpoint`、`ChatRequest.AutoCache` / `AutoCacheTTL` |
| 仅 Anthropic 有,但语义可归一 | 规范字段 + 翻译时映射 | `TopK`(OpenAI 无 `top_k`,未设置即省略) |
| 仅 Anthropic 有语义,无 OpenAI 对应值 | 原样透传为具名常量 | `FinishReasonRefusal` / `PauseTurn` / `ModelContextWindowExceeded`、`StopDetails` |
| 仅 OpenAI 有 | 规范字段,Anthropic 翻译时忽略 | `LogitBias`、`Store`、`Metadata`、`Modalities` / `Audio` |

`json:"-"` 的结构体局部字段是一条重要约定:它保证**开关永远不会泄漏到 OpenAI 形状的请求体**,同时无需为 Anthropic 单独定义一套公开请求类型。

## 3. 客户端(client.go / chat.go)

### 3.1 构造与选项

```go
client, err := aimodel.NewClient(
    aimodel.WithAPIKey("sk-..."),
    aimodel.WithBaseURL("https://api.openai.com/v1"),
    aimodel.WithProtocol(aimodel.ProtocolOpenAI),
    aimodel.WithDefaultModel(aimodel.ModelOpenaiGPT4o),
    aimodel.WithTimeout(90*time.Second),
)
```

选项一览:

| 选项 | 作用 | 备注 |
|---|---|---|
| `WithAPIKey(string)` | 鉴权密钥 | 缺失时返回 `ErrNoAPIKey` |
| `WithBaseURL(string)` | API 基址 | 自动去掉尾部 `/` |
| `WithProtocol(Protocol)` | 协议选择 | 零值 = `ProtocolOpenAI` |
| `WithDefaultModel(string)` | 默认模型 | 请求 `Model` 为空时填充 |
| `WithTimeout(time.Duration)` | HTTP 超时 | 默认 60s;**在所有选项之后统一应用**,与选项顺序无关 |
| `WithHTTPClient(*http.Client)` | 自定义 HTTP 客户端 | 传 `nil` 直接 `panic`(编程错误) |
| `WithAnthropicBeta(...string)` | `anthropic-beta` 头 | 仅 Anthropic 协议;多次调用累加,空串忽略,线上逗号连接 |
| `WithAnthropicVersion(string)` | `anthropic-version` 头 | 仅 Anthropic 协议;空串保留默认 `2023-06-01` |
| `WithAnthropicUserProfileID(string)` | `anthropic-user-profile-id` 头 | 仅 Anthropic 协议;关联终端用户档案,空串不发送该头 |

### 3.2 环境变量回退

`NewClient` 先读环境变量,再应用显式选项(**显式选项覆盖环境变量**):

| 配置 | 回退顺序 |
|---|---|
| 模型 | `AI_MODEL` |
| API Key | `AI_API_KEY` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` |
| Base URL | `AI_BASE_URL` > `OPENAI_BASE_URL` > `ANTHROPIC_BASE_URL` |

由 `GetEnv(keys ...string)` 实现:按序返回第一个非空值。

### 3.3 构造期校验

- API Key 为空 → `ErrNoAPIKey`;
- `ProtocolOpenAI` 且 Base URL 为空 → `ErrNoBaseURL`(OpenAI 兼容后端众多,无法给默认值);
- `ProtocolAnthropic` 允许 Base URL 为空 —— 请求时回退到 `https://api.anthropic.com`;
- 其它协议值 → `unsupported protocol %q`。

### 3.4 协议分派

`Client` 实现 `ChatCompleter`,`chat.go` 按 `Protocol` 做唯一的分派点:

```go
type ChatCompleter interface {
    ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error)
}
```

两条路径的共同前置动作:

1. `req.clone()` —— 深拷贝请求(切片 `Messages`/`Stop`/`Modalities`/`Tools`、映射 `LogitBias`/`Metadata`),避免 SDK 内部改写(`Stream`、默认模型)污染调用方对象;
2. 设置 `Stream` 标志;
3. `applyDefaultModel` 填充空 `Model`;
4. 流式路径下,OpenAI 侧若未设置 `StreamOptions`,自动补 `{include_usage:true}`,以便拿到末帧 usage。

## 4. 数据模型(schema.go)

### 4.1 请求 `ChatRequest`

按语义分组:

**基础**:`Model`、`Messages`、`Temperature`、`TopP`、`TopK`、`N`、`Stop`、`FrequencyPenalty`、`PresencePenalty`、`Seed`、`User`、`ResponseFormat`。

**token 上限**:
- `MaxTokens *int` —— **已弃用**。OpenAI 在 Chat Completions 上弃用了 `max_tokens`,推理模型(o 系列、GPT-5.x)会直接拒绝。
- `MaxCompletionTokens *int` —— 取代前者,上限同时覆盖可见输出与内部推理 token。

**推理/思考**:
- `ReasoningEffort string` —— 映射 OpenAI `reasoning_effort` 与 Anthropic `output_config.effort`;常量 `ReasoningEffortNone/Minimal/Low/Medium/High/XHigh`。
- `Verbosity string` —— OpenAI `verbosity`;常量 `VerbosityLow/Medium/High`。
- `Thinking *Thinking` —— `{Type, BudgetTokens(弃用), Display}`,详见 Anthropic 文档。

**工具**:`Tools []Tool`、`ToolChoice any`(`"auto"`/`"required"`/`"none"` 或 `{"type":"function","function":{"name":...}}`)、`ParallelToolCalls *bool`。

`Tool` 除 `Type` / `Function` / `CacheBreakpoint` 外,还带 Anthropic 工具定义扩展(均 `omitempty`,原样复制到 Anthropic 工具对象):`Strict *bool`、`DeferLoading *bool`、`AllowedCallers []string`、`EagerInputStreaming *bool`、`InputExamples []any`。`Type` 兼作 Anthropic 工具类型:OpenAI 的 `"function"`(及空串)表示 Anthropic 默认 custom 工具,**不发送**;其余取值(版本化内置工具)原样透传。

**流式**:`Stream bool`、`StreamOptions *StreamOptions{IncludeUsage}`。

**可观测/路由**(OpenAI):`Logprobs`、`TopLogprobs`、`LogitBias`、`ServiceTier`、`Store`、`Metadata`、`PromptCacheKey`。

**多模态**:`Modalities []string`、`Audio *AudioConfig{Voice, Format}`。

**Anthropic 直通**:`Container string`(复用服务端执行容器)、`InferenceGeo string`(数据驻留路由),均 `omitempty`。

**Anthropic 局部开关**(`json:"-"`,不出现在 OpenAI body):`AutoCache bool`、`AutoCacheTTL string`。

指针类型(`*float64` / `*int` / `*bool`)的用途是区分"未设置"与"显式设为零值" —— 例如 `Temperature=0` 与不传温度语义不同,`ParallelToolCalls=false` 才会触发 Anthropic 的 `disable_parallel_tool_use`。

### 4.2 消息与内容

```go
type Message struct {
    Role       Role       // system / user / assistant / tool
    Content    Content
    Thinking   string     `json:"reasoning_content,omitempty"`
    ToolCallID string
    ToolCalls  []ToolCall
    Audio      *MessageAudio
    CacheBreakpoint bool  `json:"-"`
}
```

`Content` 是一个**多态封装**:内部私有持有 `text string` 与 `parts []ContentPart`,通过自定义 `MarshalJSON` / `UnmarshalJSON` 在"纯字符串"与"内容块数组"两种线上形态间切换。

```go
aimodel.NewTextContent("hello")                         // → "hello"
aimodel.NewPartsContent(                                 // → [{...},{...}]
    aimodel.ContentPart{Type: "text", Text: "描述这张图"},
    aimodel.ContentPart{Type: "image_url", ImageURL: &aimodel.ImageURL{URL: dataURI}},
)
```

- `Content.Text()` —— 纯文本直接返回;多模态时拼接所有 `text` 部分;
- `Content.Parts()` —— 多模态返回内容块,纯文本返回 `nil`;
- 反序列化按首字符判别:`"` → 字符串,`[` → 数组,`null` → 双清空。

`ContentPart` 按 `Type` 选定唯一载荷:`text`→`Text`,`image_url`→`ImageURL`,`input_audio`→`InputAudio{Data,Format}`,`file`→`File{FileID | Filename+FileData}`。

### 4.3 响应

```go
type ChatResponse struct {
    ID, Object, Model string
    Created int64
    Choices []Choice
    Usage   Usage
    Error   *Error
    Container *ResponseContainer  // Anthropic 服务端执行容器
}

// ExpiresAt 保持服务端原字符串:不解析过期、不自动续期、不重试。
type ResponseContainer struct {
    ID        string
    ExpiresAt string
}

type Choice struct {
    Index        int
    Message      Message
    FinishReason FinishReason
    LogProbs     *LogProbs     // 请求设置 Logprobs 时返回
    StopDetails  *StopDetails  // Anthropic 结构化停止分类
}
```

**`FinishReason`** —— 以 OpenAI 的 `finish_reason` 为准:`stop` / `length` / `tool_calls` / `content_filter` / 遗留的 `function_call`。无 OpenAI 对应语义的 Anthropic 停止原因**原样透传**并给出具名常量,不折叠进已有值:`model_context_window_exceeded`、`refusal`、`pause_turn`。调用方应把任何非规范取值当作不透明字符串处理。

**`StopDetails`** —— `{Type, Category, Explanation}`,全部 `omitempty`,伴随 `stop_reason:"refusal"` 返回;非流式挂在 `Choice.StopDetails`,流式挂在终帧 `message_delta` 对应的 `StreamChunkChoice.StopDetails`;不存在时为 `nil`。

### 4.4 用量 `Usage`

```go
type Usage struct {
    PromptTokens, CompletionTokens, TotalTokens int
    CacheReadTokens    int  // 缓存命中读取
    CacheWriteTokens   int  // 缓存写入总量(Anthropic)
    CacheWrite5mTokens int  // 按 TTL 拆分
    CacheWrite1hTokens int
    ReasoningTokens    int  // 推理模型内部思考消耗
    ServerToolUse *ServerToolUse  // 服务端工具调用次数(Anthropic)
    InferenceGeo  string          // 实际推理地域(Anthropic)
    ServiceTier   string          // 服务等级(Anthropic)
}

type ServerToolUse struct {
    WebSearchRequests int
    WebFetchRequests  int
}
```

要点:

- **缓存读/写 token 是 `PromptTokens` 的子集**,单独暴露只为可观测性,不要重复累加计费。
- `UnmarshalJSON` 会解析嵌套结构:`prompt_tokens_details.cached_tokens` → `CacheReadTokens`,`completion_tokens_details.reasoning_tokens`(OpenAI)/ `output_tokens_details.thinking_tokens`(Anthropic)→ `ReasoningTokens`;**显式顶层字段优先**(顶层非 0 时不覆盖)。
- `CacheWrite5mTokens + CacheWrite1hTokens == CacheWriteTokens`(Anthropic 返回明细时)。
- `ServerToolUse` 的两个计数各自 `omitempty`,为 0 时从 JSON 省略;无服务端工具调用时整个对象为 `nil`。
- `Add(other)` 累加全部计数(含 `ServerToolUse`),便于跨多轮/多模型汇总;`InferenceGeo` / `ServiceTier` 描述单次请求,**不参与累加**。

## 5. 流式抽象(stream.go)

```go
type Stream struct { /* mu sync.Mutex; closed atomic.Bool; recv func() (*StreamChunk, error); usage *Usage; onClose func(*Usage) */ }

func (s *Stream) Recv() (*StreamChunk, error)  // io.EOF 表示正常结束
func (s *Stream) Usage() *Usage
func (s *Stream) Close() error                 // 幂等,可与 Recv 并发
```

设计要点:

- **协议差异被 `recv` 闭包吸收**。`Stream` 结构体本身与协议无关,OpenAI 由 `openaiRecvFunc` 构造(逐行 `data:` 解析,`[DONE]` → `io.EOF`),Anthropic 由 `anthropicRecvFunc` 构造(`event:`+`data:` 成对解析,`message_stop` → `io.EOF`)。
- **并发安全**:`Recv` 持互斥锁串行化;`Close` 用 `CompareAndSwap` 保证只执行一次,并**直接关闭底层 reader** 以解除阻塞中的 `Recv`(`http.Response.Body.Close` 可并发调用)。已关闭后 `Recv` 返回 `ErrStreamClosed`。
- **用量捕获**:任何携带 `Usage` 的 chunk 都会被记录到 `s.usage`,`Usage()` 在流结束后返回它。
- **容器 ID**:`StreamChunk.Container`(`*ResponseContainer`)在 Anthropic 流式下于读到 `message_start` 时**立即**发出一次,后续 chunk 不重复携带。流式不产出 `ChatResponse`,若等到文本 delta 才附带,仅有工具事件或随即结束的流会丢失下一轮要复用的 ID。
- **SSE 行上限** `maxStreamLineSize = 1 MB`(`bufio.Scanner` 缓冲初始 64 KB)。

### 5.1 增量合并

```go
func (m *Message) AppendDelta(delta *Message)
func (tc *ToolCall) Merge(delta *ToolCall)
```

`AppendDelta` 拼接文本与 thinking,并按 `ToolCall.Index` 就地合并工具调用(必要时用占位元素扩容切片);`Merge` 对 ID/Type/Name 采取"非空覆盖",对 `Arguments` 采取"字符串追加"(工具参数 JSON 是分片流式下发的)。

`AppendDelta` 还会按到达顺序追加 `Message.ExtraBlocks`(`[]json.RawMessage`,`json:"-"`)—— 协议未建模的内容块原文逃生口,**不解析、不合并、不改写**,由调用方自行按序处理。

### 5.2 流拦截(intercept.go / WrapStream)

两个**加性**装饰器,不改变消费侧 API:

```go
// 关闭时回调,携带最终 usage
func WrapStream(s *Stream, onClose func(*Usage)) *Stream

// 逐 chunk 观测 + 一次性完成回调
func InterceptStream(s *Stream, onChunk func(*StreamChunk), onDone func(err error)) *Stream
```

`InterceptStream` 包装 `s.recv`:每个非 nil chunk 触发 `onChunk`;`onDone` 由 `sync.Once` 保护,**恰好触发一次** —— 首个非 nil 错误(含 `io.EOF`)或 `Close`,以先到者为准。它会链式保留此前设置的 `onClose`。两者对 `s == nil` 都做了防御:立即以零值触发回调并返回 `nil`。

约束:回调必须廉价,且**不得在回调里调用 `Recv`/`Close`**(会死锁)。

## 6. 错误模型(errors.go)

**哨兵错误**:`ErrNoAPIKey`、`ErrNoBaseURL`、`ErrStreamClosed`、`ErrEmptyResponse`、`ErrNoActiveModels`。

**`APIError`** —— 承载 HTTP 状态码与服务端错误体:

```go
type APIError struct { StatusCode int; Code, Message, Type string; Err error }
```

解析策略统一为:读取响应体(上限 `maxErrorBodySize = 1 MB`)→ 尝试按协议解析错误 JSON(OpenAI `{"error":{...}}`、Anthropic `{"type":"error","error":{type,message}}`)→ **解析失败则把原始 body 塞进 `Message`**,保证信息不丢。非流式 OpenAI 路径还会检查 200 响应体里的 `error` 字段,以及 `Choices` 为空时返回 `ErrEmptyResponse`。

**`ModelError`** —— `{Model, Err}`,把错误与具体模型名关联。

**`MultiError`** —— 多模型尝试的错误集合,实现 Go 1.20+ 的 `Unwrap() []error`,因此 `errors.Is` / `errors.As` 可以匹配任意一个模型的底层错误;空集合时退化为 `ErrNoActiveModels`。

## 7. 多模型编排(composes/)

`ComposeClient` 同样实现 `aimodel.ChatCompleter`,因此**可嵌套**(一个 compose 客户端可以作为另一个的成员)。

```go
cc, err := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o",       Client: openaiClient, Weight: 3},
    {Name: "claude-opus-4", Client: anthropicClient, Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))
```

`ModelEntry.Name` 非空时覆盖请求的 `Model`(请求先按值复制,不改动调用方对象);为空则用底层客户端的默认模型。构造期校验:条目非空、每个 `Client` 非 nil。

### 7.1 选择策略

| 策略 | 行为 |
|---|---|
| `StrategyFailover`(默认) | 按定义顺序返回全部**健康**模型 |
| `StrategyRandom` | 健康模型随机打乱 |
| `StrategyWeight` | 按权重比例**无放回抽样**出完整顺序;`Weight <= 0` 视为 1 |

三者都返回一个**有序候选列表**而非单个模型 —— 分派循环依次尝试直到成功,失败转移因此对所有策略统一生效。

### 7.2 健康跟踪与恢复探测

`modelHealth` 记录 `state`(active/error)、`lastError`、`errorTime`、`errorCount`。

- 成功 → `markActive()`,清零错误计数;
- 失败 → `markError()`,`errorCount++`;
- 恢复判定 `shouldProbe`:等待时长为 **指数退避** `interval × 2^min(errorCount-1, 6)`,即最长 64 倍基础间隔(默认 60s → 最长 64 分钟)。

到期的错误模型会被 `prependRecoveryProbes` **前插**到候选列表首位,形成"优先探测,失败即继续退避"的自愈闭环。

### 7.3 上下文取消语义

分派循环在每次尝试前后检查 `ctx.Err()`:**上下文取消不污染健康状态**,直接返回 `ctx.Err()`。否则客户端主动取消会把健康模型误标为故障。

全部候选失败时返回 `*MultiError`;候选列表为空时返回 `ErrNoActiveModels`。

## 8. 模型常量(model.go)

覆盖 OpenAI、DeepSeek、Gemini、Anthropic、MiniMax、Moonshot/Kimi、智谱 GLM、豆包 Doubao、通义 Qwen 等常用模型名,纯字符串常量,仅作书写便利 —— `ChatRequest.Model` 接受任意字符串。

## 9. 目录结构

| 路径 | 内容 |
|---|---|
| 根包 `aimodel` | 客户端、schema、协议实现、流式 |
| `chat.go` / `chat_client.go` | 协议分派、`ChatCompleter` 接口与 `Protocol` 常量 |
| `schema.go` | 规范请求/响应类型 |
| `openai_chat.go` / `openai_stream.go` | OpenAI 兼容协议实现 |
| `anthropic.go` / `anthropic_chat.go` / `anthropic_stream.go` | Anthropic 类型、翻译、HTTP 与 SSE |
| `stream.go` / `intercept.go` | 流式抽象与拦截 |
| `errors.go` / `model.go` / `util.go` | 错误、模型常量、环境变量工具 |
| `composes/` | 多模型分派策略与健康跟踪 |
| `examples/` / `integrations/` | 用法示例与集成测试 |

## 10. 维护约定

官方 API 变更时,**同步更新四处**并保持一致:

1. wrapper 代码;
2. `README.md` / `CLAUDE.md`;
3. `CHANGES.md`(同步状态总账);
4. `doc/` 下对应的协议文档与变更记录。

# LLM类

## __new__
这段代码实现了 单例模式（Singleton Pattern）

这是 Python 中一种常见的设计模式，目的是确保某个类全局只有一个实例存在。
```python
_instances: Dict[str, "LLM"] = {}

def __new__(
    cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
):
    if config_name not in cls._instances:
        instance = super().__new__(cls)
        instance.__init__(config_name, llm_config)
        cls._instances[config_name] = instance
    return cls._instances[config_name]
```

在 Python 中，对象创建分为两步：

1. new(cls, ...)：分配内存空间，创建对象本身（静态方法）
2. init(self, ...)：初始化对象属性（实例方法）

一般来说我们只关心 init，但在需要控制对象创建过程的时候（比如单例模式），就需要重写 new。

Example:
```python
# 文件 A.py
llm_a = LLM("gpt-4")

# 文件 B.py
llm_b = LLM("gpt-4")
```

如果没有单例机制，这两个 llm_a 和 llm_b 是两个不同的对象，会分别创建两次连接，浪费资源。

但是有了这段 new 之后：

第一次调用 LLM("gpt-4")：

- 发现 "gpt-4" 不在 _instances 中
- 创建新的 LLM 实例并存入字典
- 返回该实例

第二次再调用 LLM("gpt-4")：

- 发现 "gpt-4" 已经存在于 _instances
- 直接返回之前的实例

所以无论在哪里调用，只要 config_name 相同，得到的就是同一个对象！

## __init__
先看这一段
```python
def __init__(
    self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
):
    if not hasattr(self, "client"):  # Only initialize if not already initialized
        llm_config = llm_config or config.llm
        llm_config = llm_config.get(config_name, llm_config["default"])
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_type = llm_config.api_type
        self.api_key = llm_config.api_key
        self.api_version = llm_config.api_version
        self.base_url = llm_config.base_url

        # Add token counting related attributes
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = (
            llm_config.max_input_tokens
            if hasattr(llm_config, "max_input_tokens")
            else None
        )
```

### Step 1
首先，如果之前没有初始化过，则从配置文件中获取 LLM 的配置信息。
```python
llm_config = llm_config or config.llm
llm_config = llm_config.get(config_name, llm_config["default"])
```

假如说我们的配置文件如下:
```toml
[llm.default]
model = "gpt-4o"
api_type = "openai"
api_key = "sk-xxx"

[llm.azure-gpt4]
model = "gpt-4-turbo"
api_type = "azure"
base_url = "https://your-resource.openai.azure.com/"
api_key = "your-key"
api_version = "2024-02-01"
```

调用
```python
llm = LLM("azure-gpt4")
```
那么就会加载 [llm.azure-gpt4] 下的所有配置。

如果没有找到，则回退到 [llm.default]。

### Step 2
设置基本属性
```python   
self.model = llm_config.model
self.max_tokens = llm_config.max_tokens
self.temperature = llm_config.temperature
self.api_type = llm_config.api_type
self.api_key = llm_config.api_key
self.api_version = llm_config.api_version
self.base_url = llm_config.base_url
```
这些都是调用模型 API 所需的基本参数。

### Step 3
添加 token 计数属性
```python
self.total_input_tokens = 0
self.total_completion_tokens = 0
self.max_input_tokens = (
    llm_config.max_input_tokens
    if hasattr(llm_config, "max_input_tokens")
    else None
)
```
用来跟踪当前会话累计消耗了多少 token，以及设置最大允许输入 token 数。

### Step 4
初始化client与模型
```python
# Initialize tokenizer
try:
    self.tokenizer = tiktoken.encoding_for_model(self.model)
except KeyError:
    # If the model is not in tiktoken's presets, use cl100k_base as default
    self.tokenizer = tiktoken.get_encoding("cl100k_base")

if self.api_type == "azure":
    self.client = AsyncAzureOpenAI(
        base_url=self.base_url,
        api_key=self.api_key,
        api_version=self.api_version,
    )
elif self.api_type == "aws":
    self.client = BedrockClient()
else:
    self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

self.token_counter = TokenCounter(self.tokenizer)
```
这一段不多说了

## 无关紧要的函数
- def count_tokens(self, text: str) -> int:
- def count_message_tokens(self, messages: List[dict]) -> int:
- def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
- def check_token_limit(self, input_tokens: int) -> bool:
- def get_limit_error_message(self, input_tokens: int) -> str:

看注释就能理解了

<details>

<summary>函数</summary>

```python
def count_tokens(self, text: str) -> int:
    """Calculate the number of tokens in a text"""
    if not text:
        return 0
    return len(self.tokenizer.encode(text))

def count_message_tokens(self, messages: List[dict]) -> int:
    return self.token_counter.count_message_tokens(messages)

def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
    """Update token counts"""
    # Only track tokens if max_input_tokens is set
    self.total_input_tokens += input_tokens
    self.total_completion_tokens += completion_tokens
    logger.info(
        f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
        f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
        f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
    )

def check_token_limit(self, input_tokens: int) -> bool:
    """Check if token limits are exceeded"""
    if self.max_input_tokens is not None:
        return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
    # If max_input_tokens is not set, always return True
    return True

def get_limit_error_message(self, input_tokens: int) -> str:
    """Generate error message for token limit exceeded"""
    if (
        self.max_input_tokens is not None
        and (self.total_input_tokens + input_tokens) > self.max_input_tokens
    ):
        return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

    return "Token limit exceeded"
```
</details>

## 函数 format_messages
作用为

把各种乱七八糟的消息格式（字典、对象、带图片的、不带图片的），统一整理成 OpenAI 接口能听懂的标准格式。

参数与返回值
```python
@staticmethod
def format_messages(
    messages: List[Union[dict, Message]], 
    supports_images: bool = False
) -> List[dict]:
```

- 支持两种信息格式
```python
#  字典形式
{"role": "user", "content": "Hello"}

#  Message 对象形式
Message.user_message("Hello")
```

都会被转成为字典形式
```python
{"role": "user", "content": "Hello"}
```
这里不重点讲Message对象

- 自动处理 base64 图像（如果模型支持）

当 supports_images=True 时，会自动把：
```python
{
    "role": "user",
    "content": "What's in this image?",
    "base64_image": "/9j/4AAQSkZJRgABAQEASABIAAD/..."
}
```
转换为：
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
            }
        }
    ]
}
```
不支持则忽略

## ⭐ 函数 ask
ask 函数的作用是向 LLM 发送一组对话消息（messages），并获取模型的回答。它支持多种特性：

|特性 | 说明 | 
| - | - |
|多种输入格式 |支持 dict 和 Message 对象
系统提示词 |	可前置插入 system 消息
流式输出 |	支持实时逐字打印
温度调节 |	控制生成创意程度
Token 限制	| 防止超出模型容量
自动重试 | 	网络错误自动恢复

### 装饰器
```python
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (OpenAIError, Exception, ValueError)
    ),  # Don't retry TokenLimitExceeded
)
```
调用一个可能会失败的方法（比如访问远程 API）时，这个装饰器会在出错后自动尝试重新执行，而不是立刻报错退出。

特别是在与 LLM（大语言模型）交互的时候，经常会遇到临时性的网络波动、超时、服务器繁忙等问题，这时候自动重试可以显著提升稳定性。

- wait
设置两次重试之间的等待时间策略：随机指数退避

第一次失败后等 1~2 秒

第二次失败后等 2~4 秒

第三次失败后等 4~8 秒

……依此类推，最多不超过 60 秒

目的是避免短时间内反复请求造成雪崩效应。

- stop
最多重试 6 次

- retry
只有在OpenAIError, Exception, ValueError这些类型的异常发生时才会重试

### 参数与返回值
```python
async def ask(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    stream: bool = True,
    temperature: Optional[float] = None,
) -> str:
```
- stream: 是否流式输出

### 预处理
```python
# Check if the model supports images
supports_images = self.model in MULTIMODAL_MODELS

# Format system and user messages with image support check
if system_msgs:
    system_msgs = self.format_messages(system_msgs, supports_images)
    messages = system_msgs + self.format_messages(messages, supports_images)
else:
    messages = self.format_messages(messages, supports_images)
```
作用：
- 判断当前模型是否支持图像（影响消息格式化）
- 格式化系统消息 + 用户消息，统一成标准 OpenAI 格式

Example:
原始输入可能是这样的：
```python
messages = [
    {"role": "user", "content": "Hello"}
]
system_msgs = [
    Message.system_message("You are a helpful assistant.")
]
```

格式化后变为
```python
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
]
```

### Token计算
```python
# Calculate input token count
input_tokens = self.count_message_tokens(messages)

# Check if token limits are exceeded
if not self.check_token_limit(input_tokens):
    error_message = self.get_limit_error_message(input_tokens)
    # Raise a special exception that won't be retried
    raise TokenLimitExceeded(error_message)
```

调用前面介绍过的 TokenCounter 计算总 token 数

若超出限制，则抛出不可重试的异常

Examples：

假设你有一个很长的历史对话，总共用了 10000 tokens，而模型上限是 8000，就会触发：

### API配置
```python
params = {
    "model": self.model,
    "messages": messages,
}

if self.model in REASONING_MODELS:
    params["max_completion_tokens"] = self.max_tokens
else:
    params["max_tokens"] = self.max_tokens
    params["temperature"] = (
        temperature if temperature is not None else self.temperature
    )
```

### 非流式请求（stream=False）
非流式（默认）	等待模型完全生成后再一次性返回全部内容
```python
if not stream:
    # Non-streaming request
    response = await self.client.chat.completions.create(
        **params, stream=False
    )

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty or invalid response from LLM")

    # Update token counts
    self.update_token_count(
        response.usage.prompt_tokens, response.usage.completion_tokens
    )

    return response.choices[0].message.content
```
非流式请求下，直接调用 OpenAI API，并返回结果。

若响应为空，则抛出异常

随后更新 token 计数

Example:

返回值
```json
{
  "choices": [
    {
      "message": {
        "content": "你好！有什么我可以帮你的吗？"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15
  }
}
```

### 流式请求（stream=True）
流式（Streaming）	模型一边生成一边逐步返回内容片段（chunks）
```python
# Streaming request, For streaming, update estimated token count before making the request
self.update_token_count(input_tokens)

response = await self.client.chat.completions.create(**params, stream=True)

collected_messages = []
completion_text = ""
async for chunk in response:
    chunk_message = chunk.choices[0].delta.content or ""
    collected_messages.append(chunk_message)
    completion_text += chunk_message
    print(chunk_message, end="", flush=True)

print()  # Newline after streaming
full_response = "".join(collected_messages).strip()
if not full_response:
    raise ValueError("Empty response from streaming LLM")

# estimate completion tokens for streaming response
completion_tokens = self.count_tokens(completion_text)
logger.info(
    f"Estimated completion tokens for streaming response: {completion_tokens}"
)
self.total_completion_tokens += completion_tokens

return full_response
```

流式请求下，先更新 token 计数，
```python
# Streaming request, For streaming, update estimated token count before making the request
self.update_token_count(input_tokens)
```

发起流式请求
```python
response = await self.client.chat.completions.create(**params, stream=True)
```

获取流式请求的返回值
```python
async for chunk in response:
    chunk_message = chunk.choices[0].delta.content or ""
    collected_messages.append(chunk_message)
    completion_text += chunk_message
    print(chunk_message, end="", flush=True)
```
- 每个 chunk 是一小段新生成的文字
- delta.content 是新增的部分
- print(..., end="", flush=True) 实现实时输出（像人在打字）


### 异常处理
```python
except TokenLimitExceeded:
    # Re-raise token limit errors without logging
    raise
except ValueError:
    logger.exception(f"Validation error")
    raise
except OpenAIError as oe:
    logger.exception(f"OpenAI API error")
    if isinstance(oe, AuthenticationError):
        logger.error("Authentication failed. Check API key.")
    elif isinstance(oe, RateLimitError):
        logger.error("Rate limit exceeded. Consider increasing retry attempts.")
    elif isinstance(oe, APIError):
        logger.error(f"API error: {oe}")
    raise
except Exception:
    logger.exception(f"Unexpected error in ask")
    raise
```

捕获异常，并记录错误信息。
捕获的异常有：
- TokenLimitExceeded: 令牌限制 exceeded
- ValueError: 值错误
- OpenAIError: OpenAI API 错误
   - AuthenticationError: 认证错误
   - RateLimitError: 速率限制 exceeded
   - APIError: API 错误
- 其他异常

## ask_with_images
这两个函数都是对ask 的扩展，用于处理图片与工具。

ask_with_images 函数用于处理图片，将图片转换为 Base64 编码，并添加到 messages 中。


```python
# Ensure the last message is from the user to attach images
if not formatted_messages or formatted_messages[-1]["role"] != "user":
    raise ValueError(
        "The last message must be from the user to attach images"
    )

# Process the last user message to include images
last_message = formatted_messages[-1]
```
确保最后一条消息是用户的，为了加入image

否则会抛出异常。


```python
# Convert content to multimodal format if needed
content = last_message["content"]
multimodal_content = (
    [{"type": "text", "text": content}]
    if isinstance(content, str)
    else content
    if isinstance(content, list)
    else []
)
```
确保内容是字符串、列表，并转换成多模态格式。

```python
# Add images to content
for image in images:
    if isinstance(image, str):
        multimodal_content.append(
            {"type": "image_url", "image_url": {"url": image}}
        )
    elif isinstance(image, dict) and "url" in image:
        multimodal_content.append({"type": "image_url", "image_url": image})
    elif isinstance(image, dict) and "image_url" in image:
        multimodal_content.append(image)
    else:
        raise ValueError(f"Unsupported image format: {image}")

# Update the message with multimodal content
last_message["content"] = multimodal_content
```

添加图片到内容中。

## aks_tool
而到了 ask_tool 这边，就禁止了流式输出
```python
async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    timeout: int = 300,
    tools: Optional[List[dict]] = None,
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
    temperature: Optional[float] = None,
    **kwargs,
) -> ChatCompletionMessage | None:
```
这边多了三个参数
- timeout: 请求超时时间
- tools: 工具列表,可供模型使用的工具列表（JSON Schema）
- tool_choice: 工具选择，控制工具使用策略（AUTO/NONE/REQUIRED/指定工具）

并进行Tool合法检验与Token计算
```python
# Validate tool_choice
if tool_choice not in TOOL_CHOICE_VALUES:
    raise ValueError(f"Invalid tool_choice: {tool_choice}")

...

# Validate tools if provided
if tools:
    for tool in tools:
        if not isinstance(tool, dict) or "type" not in tool:
            raise ValueError("Each tool must be a dict with 'type' field")
            ·
···

# If there are tools, calculate token count for tool descriptions
tools_tokens = 0
if tools:
    for tool in tools:
        tools_tokens += self.count_tokens(str(tool))

input_tokens += tools_tokens
# 工具描述本身也会占用 token，需要加入总量统计中。
```

构建新的 params 参数
```python
params = {
    "model": self.model,
    "messages": messages,
    "tools": tools,
    "tool_choice": tool_choice,
    "timeout": timeout,
    **kwargs,
}

params["stream"] = False  # 强制禁用流式（因为结构化响应难以解析）
```

返回response
```python
# Check if response is valid
if not response.choices or not response.choices[0].message:
    print(response)
    return None

return response.choices[0].message
```
返回的是一个 ChatCompletionMessage 对象，其中可能包含：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "bash",
        "arguments": "{\"command\":\"ls -l\"}"
      }
    }
  ]
}
```

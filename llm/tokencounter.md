# OpenManus详解之 LLM.py 的TokenCounter类

## TokenCounter
它的作用是：精确估算一条消息在调用大模型（如 GPT-4o、Claude）时会消耗多少 tokens。

因为：

- 大模型对输入长度有限制（比如 GPT-4o 最多 128K tokens）
- 超过限制会导致 API 报错
- 不同内容类型（文本、图片、工具调用）计算方式不同
- 图像不是按“文件大小”计费，而是按分辨率和细节级别计算
- 所以这个类就是为了解决：“我这条消息到底能不能发？”的问题。

### 常量
```python
# Token constants
BASE_MESSAGE_TOKENS = 4
FORMAT_TOKENS = 2
LOW_DETAIL_IMAGE_TOKENS = 85
HIGH_DETAIL_TILE_TOKENS = 170

# Image processing constants
MAX_SIZE = 2048
HIGH_DETAIL_TARGET_SHORT_SIDE = 768
TILE_SIZE = 512
```

含义解释：
|常量 | 说明 |
| - | - |
BASE_MESSAGE_TOKENS = 4	| 每条消息的基础开销（角色、括号等结构）
FORMAT_TOKENS = 2	| 整个消息列表的格式开销（如 [, ]）
LOW_DETAIL_IMAGE_TOKENS = 85|	低清图固定收费 85 tokens
HIGH_DETAIL_TILE_TOKENS = 170|	每个 512×512 的高清图块约 170 tokens
MAX_SIZE = 2048	| 图像先缩放到不超过 2048px 边长
HIGH_DETAIL_TARGET_SHORT_SIDE = 768	|短边统一拉到 768px
TILE_SIZE = 512	 | 切成 512px 正方形小块

### 函数 count_text(self, text: str) -> int
使用tokenizer计算字符串的token数量
```python
def count_text(self, text: str) -> int:
    """Calculate tokens for a text string"""
    return 0 if not text else len(self.tokenizer.encode(text))
```

Example:
```python
tokencounter = TokenCounter()  # 初始化一个 LLM 实例

text = "Hello, how are you?"
tokens = tokencounter.token_counter(text)
print(tokens)  # 输出: 7
# tokens: ['Hello', ',', ' how', ' are', ' you', '?']
```

### 函数 count_image(image_item: dict) -> int
根据图像的 detail 设置和尺寸，返回所需 tokens。

假设我们输入的image_item是一个字典，大概以下三个样
```python
image_item = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,..."},
    "detail": "low"
}
###################################################
image_item = {
    "type": "image_url",
    "image_url": {"url": "..."},
    "detail": "high",
    "dimensions": [1920, 1080]
}
####################################################
image_item = {
    "type": "image_url",
    "image_url": {"url": "..."},
    "detail": "high"
    # 没有 dimensions 字段
}
```

首先，我们从image_item里获取"detail"字段

如果"detail"字段为"low"，则返回 85
```python
detail = image_item.get("detail", "medium")

# For low detail, always return fixed token count
if detail == "low":
    return self.LOW_DETAIL_IMAGE_TOKENS
```

如果"detail"字段为"high" 或 "medium"，进行以下判断

如果有"dimensions"字段，则返回该字段的值，并输入到辅助函数中
```python
# For medium detail (default in OpenAI), use high detail calculation
# OpenAI doesn't specify a separate calculation for medium

# For high detail, calculate based on dimensions if available
if detail == "high" or detail == "medium":
    # If dimensions are provided in the image_item
    if "dimensions" in image_item:
        width, height = image_item["dimensions"]
        return self._calculate_high_detail_tokens(width, height)
```

没有，则默认按 1024 算
```python
return (
    self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
)
```

完整代码
```python
def count_image(self, image_item: dict) -> int:
    """
    Calculate tokens for an image based on detail level and dimensions

    For "low" detail: fixed 85 tokens
    For "high" detail:
    1. Scale to fit in 2048x2048 square
    2. Scale shortest side to 768px
    3. Count 512px tiles (170 tokens each)
    4. Add 85 tokens
    """
    # Get detail level
    detail = image_item.get("detail", "medium")

    # For low detail, always return fixed token count
    if detail == "low":
        return self.LOW_DETAIL_IMAGE_TOKENS

    # For medium detail (default in OpenAI), use high detail calculation
    # OpenAI doesn't specify a separate calculation for medium

    # For high detail, calculate based on dimensions if available
    if detail == "high" or detail == "medium":
        # If dimensions are provided in the image_item
        if "dimensions" in image_item:
            width, height = image_item["dimensions"]
            return self._calculate_high_detail_tokens(width, height)

    return (
        self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
    )

```

### 辅助函数 _calculate_high_detail_tokens(self, width: int, height: int) -> int
这个函数用于计算高细节图片的token数。具体实现如下：

1. 缩放到最大2048px内
如果当前长或宽高于 2048px，则将长或宽缩放到 2048px。
```python
# Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
if width > self.MAX_SIZE or height > self.MAX_SIZE:
    scale = self.MAX_SIZE / max(width, height)
    width = int(width * scale)
    height = int(height * scale)
```

2. 短边缩放到 768 px
```python
# Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
scaled_width = int(width * scale)
scaled_height = int(height * scale)
```

3. 切成 512 px 的小块
```python
# Step 3: Count number of 512px tiles
tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
total_tiles = tiles_x * tiles_y
```

4. 计算最终的 Token
```python
# Step 4: Calculate final token count
return (
    total_tiles * self.HIGH_DETAIL_TILE_TOKENS
) + self.LOW_DETAIL_IMAGE_TOKENS
```

Example:
```python
image_item = {
    "type": "image_url",
    "image_url": {"url": "..."},
    "detail": "high",
    "dimensions": [1920, 1080]
}

tokens = tokencounter.count_image(image_item)
# → 调用 _calculate_high_detail_tokens(1920, 1080)

"""
Step 1: 缩放到最大 2048px 内 ✅
当前最大边是 1920 < 2048 → 不缩放

Step 2: 短边缩放到 768px
当前短边是 1080 → scale = 768 / 1080 ≈ 0.711
新宽高：
width = 1920 × 0.711 ≈ 1365
height = 1080 × 0.711 ≈ 768

Step 3: 切成 512px 小块
x方向：ceil(1365 / 512) = ceil(2.66) = 3
y方向：ceil(768 / 512) = ceil(1.5) = 2
总 tile 数 = 3 × 2 = 6

Step 4: 计算最终 token
total_tokens = 6 * 170 + 85 = 1020 + 85 = 1105
"""
```

完整代码
```python
def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
    """Calculate tokens for high detail images based on dimensions"""
    # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
    if width > self.MAX_SIZE or height > self.MAX_SIZE:
        scale = self.MAX_SIZE / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
    scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    # Step 3: Count number of 512px tiles
    tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
    tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
    total_tiles = tiles_x * tiles_y

    # Step 4: Calculate final token count
    return (
        total_tiles * self.HIGH_DETAIL_TILE_TOKENS
    ) + self.LOW_DETAIL_IMAGE_TOKENS
```

### 函数 count_content(self, content: Union[str, List[Union[str, dict]]]) -> int
处理多模态消息内容

代码
```python
def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
    """Calculate tokens for message content"""
    # content为空，返回0
    if not content:
        return 0
    
    # 如果为 str 格式，直接返回 count_text 函数处理结果
    if isinstance(content, str):
        return self.count_text(content)
    
    # 如果为 list 列表或其他格式，遍历列表，对每个元素进行处理
    token_count = 0
    for item in content:
        # 元素为 str ，同上
        if isinstance(item, str):
            token_count += self.count_text(item)
        # 元素为 dict ，判断是否存在 "text" 或 "content" 键，存在则处理
        elif isinstance(item, dict):
            if "text" in item:
                token_count += self.count_text(item["text"])
            elif "image_url" in item:
                token_count += self.count_image(item)
    return token_count
```

Example:
```python
content = [
    "What's in this picture?",
    {
        "type": "image_url",
        "image_url": {"url": "..."},
        "detail": "high",
        "dimensions": [1920, 1080]
    }
]

tokencounter = TokenCounter()
tokens = tokencounter.count_content(content)
```

### 函数 count_tool_calls(self, tool_calls: List[dict]) -> int
计算 AI 调用外部工具（如 bash、search）所需的 token。

源码
```python
def count_tool_calls(self, tool_calls: List[dict]) -> int:
    """Calculate tokens for tool calls"""
    token_count = 0
    # 遍历工具调用列表
    for tool_call in tool_calls:
        # 是否包含"function"键
        if "function" in tool_call:
            function = tool_call["function"]
            # 获取函数名称并计算token数
            token_count += self.count_text(function.get("name", ""))
            # 获取函数参数并计算token数
            token_count += self.count_text(function.get("arguments", ""))
    return token_count
```

Example:
```python
tool_calls = [
    {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": "{\"command\": \"ls -l\"}"
        }
    }
]

tokencounter = TokenCounter()
token_count = tokencounter.count_tool_calls(tool_calls)
# = count_text("bash") + count_text('{"command": "ls -l"}')
# ≈ 1 + 8 = 9 tokens
```

### 函数 count_message_tokens(self, messages: List[dict]) -> int
函数用于计算给定消息列表的 token 数量(计算整个对话历史的总 token 数量)。

代码
```python
def count_message_tokens(self, messages: List[dict]) -> int:
    """Calculate the total number of tokens in a message list"""
    total_tokens = self.FORMAT_TOKENS  # Base format tokens

    for message in messages:
        # 先加上Base token
        tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

        # 加上角色的Token
        tokens += self.count_text(message.get("role", ""))

        # 加上对话内容的Token
        if "content" in message:
            tokens += self.count_content(message["content"])

        # 加上工具调用的Token
        if "tool_calls" in message:
            tokens += self.count_tool_calls(message["tool_calls"])

        # 加上工具名称与调用ID的Token
        tokens += self.count_text(message.get("name", ""))
        tokens += self.count_text(message.get("tool_call_id", ""))

        total_tokens += tokens

    return total_tokens
```

Example:
```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            "What's in this image?",
            {
                "type": "image_url",
                "image_url": {"url": "..."},
                "detail": "high",
                "dimensions": [1920, 1080]
            }
        ]
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\": \"ls -l\"}"
                }
            }
        ]
    }
]

tokencounter = TokenCounter()
total_tokens = tokencounter.count_tokens(messages)
```

这里偷懒，问AI了

我们来一步一步加：

1️⃣ FORMAT_TOKENS = 2

2️⃣ 第一条消息：system
- base: 4
- role (system): ≈2
- content (You are...): ≈7
- total: 4+2+7 = 13

3️⃣ 第二条消息：user（图文混合）
- base: 4
- role (user): ≈2
- content:
- text: "What's in this image?" ≈ 6
- image: 1920x1080 high detail → 1105
- total: 4+2+6+1105 = 1117

4️⃣ 第三条消息：assistant（tool call）
- base: 4
- role (assistant): ≈3
- tool_calls: 9（前面算过）
- total: 4+3+9 = 16

✅ 最终总计：
```text
total_tokens = FORMAT_TOKENS (2)
             + system_msg (13)
             + user_msg (1117)
             + assistant_msg (16)
             ---------------------
             = 1148 tokens
```

## 完整代码
```python
class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        # Get detail level
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens
```

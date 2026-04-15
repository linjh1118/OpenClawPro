# Memory Management Algorithms — Research Recipe

本文档提取自 [nousresearch/hermes-agent](https://github.com/nousresearch/hermes-agent) 的记忆管理系统，精炼出对科研有价值的算法创新点，可作为独立 recipe 复用。

---

## 目录

1. [HRR: 全息缩减表示](#1-hrr-全息缩减表示)
2. [三路混合检索](#2-三路混合检索)
3. [非对称信任评分](#3-非对称信任评分)
4. [时序衰减评分](#4-时序衰减评分)
5. [上下文迭代压缩](#5-上下文迭代压缩)
6. [系统提示冻结快照](#6-系统提示冻结快照)
7. [矛盾检测](#7-矛盾检测)
8. [多实体代数推理](#8-多实体代数推理)

---

## 1. HRR: 全息缩减表示

**来源**: `plugins/memory/holographic/holographic.py`

**核心思想**: 用固定维度的相位向量编码离散概念，通过向量代数实现存储、检索和推理。信息以分布式方式压缩存储在向量中，检索时通过解绑操作恢复原始概念。

### 1.1 向量生成 (确定性相位编码)

```python
import numpy as np
import hashlib

def generate_atom(seed: str, dim: int = 1024) -> np.ndarray:
    """从种子生成确定性相位向量，值域 [0, 2π)"""
    h = hashlib.sha256(seed.encode()).digest()
    # 取前 dim 个字节映射到 [0, 2π)
    phases = np.frombuffer(h * (dim // 32 + 1), dtype=np.uint8)[:dim]
    return (phases / 255.0) * 2 * np.pi

def hrr_vector(text: str, dim: int = 1024) -> np.ndarray:
    """从文本生成相位向量 (单词 bag-of-words 编码)"""
    words = text.lower().split()
    vectors = [generate_atom(w, dim) for w in words]
    if not vectors:
        return np.zeros(dim)
    # 环形均值叠加
    x = np.mean([np.cos(v) for v in vectors], axis=0)
    y = np.mean([np.sin(v) for v in vectors], axis=0)
    return np.arctan2(y, x) % (2 * np.pi)
```

### 1.2 三种基本运算

```python
def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """绑定两个概念 = 循环卷积 (相位相加)
    用于存储 (key, value) 关联"""
    return (a + b) % (2 * np.pi)

def unbind(memory: np.ndarray, key: np.ndarray) -> np.ndarray:
    """解绑检索 = 循环相关 (相位相减)
    从 memory 中恢复 key 对应的 value"""
    return (memory - key) % (2 * np.pi)

def bundle(*vectors: np.ndarray) -> np.ndarray:
    """捆绑合并 = 环形均值叠加
    用于合并多个概念为一个分布式表示"""
    x = np.mean([np.cos(v) for v in vectors], axis=0)
    y = np.mean([np.sin(v) for v in vectors], axis=0)
    return np.arctan2(y, x) % (2 * np.pi)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """相位余弦相似度，用于比较两个 HRR 向量"""
    return float(np.mean(np.cos(a - b)))
```

### 1.3 事实存储与检索

```python
class FactStore:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.memory_bank = np.zeros(dim)  # 捆绑所有事实的全局记忆库
        self.facts: dict[str, dict] = {}

    def add_fact(self, content: str, entities: list[str], category: str = "general"):
        """存储事实：将 content 和 entities 绑定到记忆中"""
        content_vec = hrr_vector(content, self.dim)
        entity_vec = bundle(*[hrr_vector(e, self.dim) for e in entities])

        role_content = generate_atom("ROLE_CONTENT", self.dim)
        role_entity = generate_atom("ROLE_ENTITY", self.dim)

        fact_vec = bundle(
            bind(role_content, content_vec),
            bind(role_entity, entity_vec)
        )

        self.memory_bank = bundle(self.memory_bank, fact_vec)
        self.facts[content] = {
            "content": content,
            "entities": entities,
            "category": category,
            "vector": fact_vec,
            "trust": 0.5,  # 见第 3 节
            "created_at": None,  # datetime
        }

    def probe(self, entity: str) -> list[str]:
        """通过解绑操作检索与某实体相关的所有事实"""
        role_entity = generate_atom("ROLE_ENTITY", self.dim)
        entity_vec = hrr_vector(entity, self.dim)
        recovered = unbind(self.memory_bank, bind(role_entity, entity_vec))

        # 在所有事实中找与 recovered 最相似的
        scores = [(f, cosine_similarity(f["vector"], recovered)) for f in self.facts.values()]
        return [f["content"] for f, s in sorted(scores, key=lambda x: -x[1])]
```

**创新点**: 与传统向量数据库 (cosine similarity on embeddings) 不同，HRR 支持**代数操作** — 绑定可以表达 `(key, value)` 关联，解绑可以结构化检索，捆绑支持分布式存储。所有操作都是确定性的、无需训练。

---

## 2. 三路混合检索

**来源**: `plugins/memory/holographic/retrieval.py:48-112`

**核心思想**: 融合关键词检索 (FTS5)、词法相似度 (Jaccard) 和语义结构相似度 (HRR)，三种信号互补，最后叠加信任分和时序衰减。

### 2.1 核心评分公式

```
final_score(i) = relevance(i) × trust(i) × decay(i)

其中:
  relevance(i) = 0.4 × FTS5_norm(i) + 0.3 × Jaccard(i) + 0.3 × HRR_sim(i)
  trust(i)     = 事实的累积信任分 ∈ [0.0, 1.0]          (见第 3 节)
  decay(i)     = 0.5^(age_days / half_life)              (见第 4 节)
```

### 2.2 简化实现

```python
def jaccard_similarity(query: str, text: str) -> float:
    """词法 Jaccard 相似度"""
    def tokens(s):
        return set(s.lower().split())
    a, b = tokens(query), tokens(text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def hybrid_search(
    query: str,
    facts: list[dict],
    w_fts: float = 0.4,
    w_jac: float = 0.3,
    w_hrr: float = 0.3,
    half_life_days: float = 30.0,
) -> list[tuple[dict, float]]:
    """三路混合检索 + 信任加权 + 时序衰减"""

    query_vec = hrr_vector(query)

    # 1. 预计算 query 词集
    query_tokens = set(query.lower().split())

    scored = []
    for fact in facts:
        # FTS5: 假设已通过外部 FTS5 引擎获得排名分数 (0-1 归一化)
        fts_score = fact.get("fts_score", 0.0)

        # Jaccard: query vs content + tags
        tags_text = " ".join(fact.get("tags", []))
        jac = jaccard_similarity(query, fact["content"] + " " + tags_text)

        # HRR: 向量相似度
        fact_vec = fact.get("vector")
        hrr_sim = cosine_similarity(query_vec, fact_vec) if fact_vec is not None else 0.0

        # 归一化: HRR 从 [-1,1] 移到 [0,1]
        hrr_sim = (hrr_sim + 1) / 2

        # 加权组合
        relevance = w_fts * fts_score + w_jac * jac + w_hrr * hrr_sim

        # 信任加权
        trust = fact.get("trust", 0.5)

        # 时序衰减
        if "created_at" in fact and fact["created_at"]:
            age_days = (datetime.now() - fact["created_at"]).days
            decay = 0.5 ** (age_days / half_life_days)
        else:
            decay = 1.0

        score = relevance * trust * decay
        scored.append((fact, score))

    # 排序返回
    return sorted(scored, key=lambda x: -x[1])
```

### 2.3 退化策略 (无 numpy 时)

```python
# 如果 HRR 不可用，重分配权重:
w_fts, w_jac, w_hrr = 0.6, 0.4, 0.0
# 语义能力降级但仍保持多路检索框架
```

**创新点**: 三种检索信号分别捕捉不同维度 — FTS5 捕捉精确关键词匹配，Jaccard 捕捉词法覆盖度，HRR 捕捉语义结构相似度。三者加权和 + 可插拔的信任/衰减机制提供了极大的灵活性。

---

## 3. 非对称信任评分

**来源**: `plugins/memory/holographic/store.py:78-82`

**核心思想**: 负面反馈的惩罚力度是正面反馈奖励力度的 2 倍，形成不对称奖惩机制。直觉是：错误记忆的危害远大于错过一条好记忆。

```python
TRUST_INITIAL = 0.5
TRUST_POSITIVE_DELTA = +0.05    # 正面反馈: +5%
TRUST_NEGATIVE_DELTA = -0.10    # 负面反馈: -10% (2x 惩罚)
TRUST_MIN = 0.0
TRUST_MAX = 1.0
TRUST_EXCLUDE_THRESHOLD = 0.3   # 信任 < 0.3 则不参与检索

def update_trust(current: float, is_positive: bool) -> float:
    """非对称更新信任分"""
    delta = TRUST_POSITIVE_DELTA if is_positive else TRUST_NEGATIVE_DELTA
    return max(TRUST_MIN, min(TRUST_MAX, current + delta))

def is_retrievable(trust: float) -> bool:
    """信任低于阈值的记忆不参与检索"""
    return trust >= TRUST_EXCLUDE_THRESHOLD
```

**科研价值**: 这种非对称设计可应用于知识库清理、RLHF 奖励 shaping、信念更新等领域。直觉可形式化为：假阳性 (false positive) 成本 > 假阴性 (false negative) 成本。

---

## 4. 时序衰减评分

**来源**: `plugins/memory/holographic/retrieval.py:569-593`

**核心思想**: 记忆的相关性随时间指数衰减，半衰期可配置。默认半衰期 30 天。

```python
def temporal_decay(age_days: float, half_life_days: float = 30.0) -> float:
    """指数衰减: 每经过 half_life_days，记忆权重减半"""
    return 0.5 ** (age_days / half_life_days)

# 使用示例
from datetime import datetime

age_days = (datetime.now() - fact["created_at"]).days
decay = temporal_decay(age_days, half_life_days=30.0)
score = base_score * decay
```

**变体**:
- 不同类别使用不同半衰期: e.g., 个人偏好 (60天), 技术事实 (180天), 闲聊 (7天)
- 使用 sigmoid 替代指数: `decay = 1 / (1 + exp(k * (age - threshold)))`

---

## 5. 上下文迭代压缩

**来源**: `agent/context_compressor.py:314-436`

**核心思想**: 当上下文窗口即将溢出时，不是简单截断，而是用 LLM 生成结构化摘要，并跨多次压缩周期**迭代更新**摘要。

### 5.1 压缩触发条件

```python
def should_compress(
    prompt_tokens: int,
    context_length: int = 200_000,
    threshold_pct: float = 0.50,
    min_tokens: int = 30_000,
) -> bool:
    """超过阈值 tokens 时触发压缩"""
    threshold = max(context_length * threshold_pct, min_tokens)
    return prompt_tokens >= threshold
```

### 5.2 摘要生成提示模板

```python
SUMMARY_PROMPT = """You are a summarization agent. Do NOT respond to any questions or requests.
Summarize the following conversation turns concisely.

{previous_summary}

---
NEW TURNS (since last summary):
{new_turns}

---
Produce a structured summary with these sections:

## Goal
What was the user trying to accomplish?

## Constraints
Key requirements, rules, or limitations.

## Progress
- Done: What has been completed?
- In Progress: What is currently being worked on?
- Blocked: What is stuck or waiting?

## Key Decisions
Important choices made and their rationale.

## Resolved Questions
Questions that were answered (prevent model from re-answering).

## Pending User Asks
Questions the user has asked but not yet answered.

## Relevant Files
Files mentioned or modified.

## Remaining Work
What still needs to be done.

## Critical Context
Anything essential that must not be lost.

## Tools & Patterns
Tool usage patterns that worked well.

Keep the summary under {budget_tokens} tokens.
"""
```

### 5.3 迭代压缩算法

```python
def compress_context(
    messages: list[dict],
    previous_summary: str | None = None,
    protect_first_n: int = 3,
    protect_last_n: int = 6,
    summary_budget: int = 2000,
) -> list[dict]:
    """
    1. 保护头部 (系统提示 + 前几次交互)
    2. 保护尾部 (最近 N 条, 按 token 预算)
    3. 对中间部分生成摘要
    4. 如果已有摘要, 增量更新
    """

    # Step 1: 头部保护
    protected_head = messages[:protect_first_n]

    # Step 2: 尾部保护 (略过实现, 假设已有 _find_tail_by_tokens)
    protected_tail = messages[-protect_last_n:]

    # Step 3: 中间部分 (待压缩)
    middle = messages[protect_first_n:-protect_last_n]
    if not middle:
        return messages  # 无需压缩

    # Step 4: 构建摘要
    new_turns_text = _format_turns(middle)
    prompt = SUMMARY_PROMPT.format(
        previous_summary=previous_summary or "(No previous summary)",
        new_turns=new_turns_text,
        budget_tokens=summary_budget,
    )

    summary = call_llm(prompt)  # 使用廉价模型

    # Step 5: 组装新消息列表
    if previous_summary is None:
        # 首次压缩: head + summary + tail
        return protected_head + [summary] + protected_tail
    else:
        # 迭代更新: 用新摘要替换旧摘要 + tail
        # (需要找到并替换旧摘要消息)
        result = protected_head + [summary] + protected_tail
        return result
```

### 5.4 摘要预算自适应

```python
def compute_summary_budget(
    content_tokens: int,
    context_length: int = 200_000,
    min_budget: int = 2000,
    max_budget: int = 12_000,
) -> int:
    """
    摘要预算 = min(max(2000, 20% of content), 5% of context_length)
    """
    return int(max(min_budget, min(
        content_tokens * 0.20,
        context_length * 0.05,
        max_budget
    )))
```

**创新点**: 迭代摘要 (iterative summarization) 保留了历史上下文的压缩版本，而不是简单的滑动窗口。这使得模型可以跨多个压缩周期积累"机构记忆"。相比 RAG 的向量检索，这种方法更适合**过程性记忆**（任务进度、工作流程）而非事实性检索。

---

## 6. 系统提示冻结快照

**来源**: `tools/memory_tool.py:132-135`

**核心思想**: 在会话开始时一次性加载所有配置/记忆并冻结到系统提示中，会话期间即使底层文件被修改也不更新系统提示。保护 LLM 的 **KV Cache / Prefix Cache**，大幅降低每轮推理成本。

```python
class MemoryStore:
    def __init__(self, memory_file: str, user_file: str):
        self.memory_file = memory_file
        self.user_file = user_file
        self._snapshot: dict | None = None
        self._loaded = False

    def load_snapshot(self) -> None:
        """会话开始时: 一次性加载并冻结"""
        if self._loaded:
            return
        self._snapshot = {
            "memory": self._read_file(self.memory_file),
            "user": self._read_file(self.user_file),
        }
        self._loaded = True

    def get_system_prompt_content(self) -> dict:
        """返回冻结快照 (会话期间不变)"""
        if not self._loaded:
            self.load_snapshot()
        return self._snapshot

    def write_memory(self, entry: str) -> None:
        """写入新记忆 (文件被更新, 但快照不变)"""
        self._append_to_file(self.memory_file, entry)
        # 注意: 不更新 self._snapshot!

    def format_for_system_prompt(self) -> str:
        """格式化为系统提示词片段"""
        return SYSTEM_PROMPT_TEMPLATE.format(
            memory=self._snapshot["memory"],
            user=self._snapshot["user"],
        )
```

**科研价值**: 这是一个**缓存一致性**问题在 LLM Agent 场景下的工程解法。冻结快照避免了在 KV Cache 中反复修改 prefix 导致的重新计算开销，同时允许后台持久化写入。在任何需要同时兼顾**一致性**和**性能**的场景下都可借鉴。

---

## 7. 矛盾检测

**来源**: `plugins/memory/holographic/retrieval.py:338-442`

**核心思想**: 找到知识库中"实体重叠但内容相悖"的事实对。自直觉：如果两个事实涉及同一实体群，但 HRR 内容向量差异很大，则可能存在矛盾。

```python
def detect_contradictions(
    facts: list[dict],
    entity_overlap_threshold: float = 0.3,
    content_similarity_threshold: float = 0.5,
) -> list[tuple[dict, dict, float]]:
    """
    矛盾检测算法:
    1. 找实体重叠度高的事实对 (Jaccard >= threshold)
    2. 检查内容向量相似度是否低于 threshold
    3. 矛盾分数 = 实体重叠度 × (1 - 内容相似度)
    """
    results = []
    n = len(facts)

    for i in range(n):
        for j in range(i + 1, min(n, i + 500)):  # O(n²) 裁剪到 500 对
            f1, f2 = facts[i], facts[j]

            # 实体 Jaccard
            e1, e2 = set(f1["entities"]), set(f2["entities"])
            if not e1 or not e2:
                continue
            entity_jacc = len(e1 & e2) / len(e1 | e2)

            if entity_jacc < entity_overlap_threshold:
                continue

            # 内容向量相似度
            if f1.get("vector") is None or f2.get("vector") is None:
                continue
            content_sim = cosine_similarity(f1["vector"], f2["vector"])

            if content_sim >= content_similarity_threshold:
                continue  # 内容相似, 不矛盾

            # 矛盾分数
            score = entity_jacc * (1 - content_sim)
            results.append((f1, f2, score))

    return sorted(results, key=lambda x: -x[2])
```

**科研价值**: 可用于构建**主动式知识库审计**管道。在任何需要维护多源知识一致性的场景（医疗、法律、金融）中，定期运行矛盾检测可以提前发现知识退化问题。

---

## 8. 多实体代数推理

**来源**: `plugins/memory/holographic/retrieval.py:260-336`

**核心思想**: 使用 HRR 代数查询"同时涉及多个实体的记忆"。AND 语义：取实体在每条事实中"存在感"的最小值。

```python
def reason(facts: list[dict], entities: list[str]) -> list[tuple[dict, float]]:
    """
    多实体推理: 找出同时涉及所有输入实体的记忆

    算法:
    1. 对每个实体, 在每条事实中计算该实体的"存在感"分数
       → 即 fact.vector 与 bind(ROLE_ENTITY, entity_vec) 的相似度
    2. 每条事实取所有实体中的最小存在感 (AND 语义)
    3. 按分数排序
    """
    role_entity = generate_atom("ROLE_ENTITY")

    results = []
    for fact in facts:
        scores = []
        for entity in entities:
            entity_vec = hrr_vector(entity)
            # 解绑出该实体在 fact 中的"绑定表示"
            bound = bind(role_entity, entity_vec)
            # 计算 fact 中该实体的存在度
            presence = cosine_similarity(fact["vector"], bound)
            scores.append(presence)

        # AND: 取最小 (必须所有实体都存在)
        min_score = min(scores)
        results.append((fact, min_score))

    return sorted(results, key=lambda x: -x[1])
```

**科研价值**: HRR 代数为知识推理提供了**可组合的查询语言**：probe (单实体检索) = `unbind(memory, entity)`，reason (多实体 AND) = `min(unbind(...))`。这比简单的向量相似度阈值更表达力强，且计算开销与简单 cosine 检索相当。

---

## 算法对照表

| 算法 | 输入 | 输出 | 核心创新 | 适用场景 |
|------|------|------|---------|---------|
| HRR 向量编码 | 文本 | 相位向量 | 绑定/解绑/捆绑代数运算 | 结构化知识存储、推理 |
| 三路混合检索 | query + facts | 排序列表 | FTS5+Jaccard+HRR 加权融合 | 记忆检索 |
| 非对称信任 | 反馈信号 | 信任分数 | 2:1 惩罚/奖励比 | 知识库质量控制 |
| 时序衰减 | age_days | decay因子 | 指数半衰期衰减 | 记忆时效性管理 |
| 迭代压缩 | messages | summary + messages | 跨周期摘要迭代更新 | 长程任务上下文管理 |
| 冻结快照 | memory files | system prompt | 写时更新、读时冻结 | LLM 缓存优化 |
| 矛盾检测 | facts pairs | 矛盾对列表 | 实体重叠 × 内容分歧 | 知识库审计 |
| 代数推理 | entities list | facts sorted | HRR AND 语义查询 | 多约束知识检索 |

---

## 快速启动: 运行一个完整示例

```python
import numpy as np
from datetime import datetime, timedelta

# 初始化
dim = 1024

# 1. 添加记忆
store = FactStore(dim=dim)
store.add_fact("用户偏好使用 tmux 管理终端会话", ["tmux", "用户偏好"])
store.add_fact("项目使用 Python 3.11 和 FastAPI", ["Python", "FastAPI", "项目"])
store.add_fact("数据库连接使用环境变量 DATABASE_URL", ["数据库", "环境变量"])
store.add_fact("用户偏好使用 Darkstar 配色方案", ["配色", "用户偏好"])

# 2. 模拟时间流逝 (部分记忆变旧)
store.facts["用户偏好使用 Darkstar 配色方案"]["created_at"] = \
    datetime.now() - timedelta(days=60)

# 3. 模拟反馈
store.facts["用户偏好使用 Darkstar 配色方案"]["trust"] = \
    update_trust(store.facts["用户偏好使用 Darkstar 配色方案"]["trust"], is_positive=False)

# 4. 检索
results = hybrid_search(
    query="用户的编辑器配色和终端偏好是什么?",
    facts=list(store.facts.values()),
    half_life_days=30.0,
)
for fact, score in results[:3]:
    print(f"[{score:.3f}] {fact['content']}")
```

预期输出: 配色记忆因时间衰减和负面信任，排名会显著下降。

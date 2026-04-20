"""
EpisodicMemoryStore - Core memory storage and retrieval.
"""

import logging
import time
import uuid
from typing import List, Optional

from .config import MemoryConfig
from .item import MemoryItem
from .policy import RetrievalPolicy, should_write_to_memory


class EpisodicMemoryStore:
    """Task-local episodic memory store.

    Stores tool results and errors to help the agent remember
    context across iterations within a single task.
    """

    _logger = logging.getLogger("memory.store")
    _embedding_model = None
    _embedding_tokenizer = None
    _bm25_model = None
    _embedding_available = None  # None=unknown, True=available, False=unavailable

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._items: List[MemoryItem] = []
        self._iteration = 0
        self._enabled = config.enabled
        self._item_embeddings: dict[str, List[float]] = {}  # item_id -> embedding
        self._embedding_cache: dict[str, List[float]] = {}  # text -> embedding cache
        self._bm25_cache: dict[str, float] = {}  # item_id -> bm25_score for current query

    def reset(self) -> None:
        """Reset memory for new task."""
        self._items = []
        self._iteration = 0
        self._item_embeddings.clear()
        self._embedding_cache.clear()
        self._bm25_cache.clear()
        self._logger.debug("[EpisodicMemoryStore] Reset for new task")

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self._iteration += 1

    def write(
        self,
        content: str,
        source: str,
        source_detail: str = "",
        memory_type: str = "result",
    ) -> Optional[MemoryItem]:
        """Write a memory item.

        Args:
            content: The content to store
            source: Source type ("tool_result", "error", "user_prompt")
            source_detail: Additional source info (e.g., tool name)
            memory_type: Memory type ("fact", "instruction", "error", "result")

        Returns:
            The created MemoryItem, or None if not written
        """
        if not self._enabled:
            return None

        # Check budget
        if len(self._items) >= self.config.max_items:
            # Remove oldest item to make space
            self._items.pop(0)
            self._logger.debug(f"[EpisodicMemoryStore] Budget exceeded, removed oldest item")

        item = MemoryItem(
            id=str(uuid.uuid4())[:8],
            content=content[:2000],  # Truncate very long content
            source=source,
            source_detail=source_detail,
            iteration=self._iteration,
            memory_type=memory_type,
            created_at=time.time(),
        )
        self._items.append(item)

        # Compute and cache embedding for SEMANTIC and HYBRID retrieval
        policy_val = self.config.retrieval_policy.value
        if policy_val in ("semantic", "hybrid"):
            self._compute_item_embedding(item)

        self._logger.debug(f"[EpisodicMemoryStore] Wrote item: {item.id} ({source}, {memory_type})")
        return item

    def retrieve(self, max_items: Optional[int] = None, query: Optional[str] = None) -> List[MemoryItem]:
        """Retrieve relevant memory items.

        Args:
            max_items: Maximum items to retrieve (default from config)
            query: Query string for SEMANTIC retrieval (required if policy is SEMANTIC)

        Returns:
            List of retrieved MemoryItems sorted by policy-specific score
        """
        if not self._enabled or not self._items:
            return []

        if max_items is None:
            max_items = self.config.retrieval_max

        current_time = time.time()

        # Filter by trust threshold
        filtered_items = [item for item in self._items if item.trust >= self.config.trust_exclude_threshold]

        # Pre-compute BM25 scores for HYBRID/SEMANTIC policies
        if query and self.config.retrieval_policy.value in ("hybrid", "semantic"):
            self._compute_bm25_scores(query, filtered_items)

        # Score items by retrieval policy
        scored_items = []
        for item in filtered_items:
            score = self._score_by_policy(item, current_time, query)
            scored_items.append((item, score))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Touch accessed items and return top items
        retrieved = []
        for item, _ in scored_items[:max_items]:
            item.touch(current_time)
            retrieved.append(item)

        self._logger.debug(f"[EpisodicMemoryStore] Retrieved {len(retrieved)} items (policy={self.config.retrieval_policy.value})")
        return retrieved

    def _score_by_policy(self, item: MemoryItem, current_time: float, query: Optional[str] = None) -> float:
        """Calculate retrieval score based on the configured policy.

        RECENT:     score = recency_score × trust_score × decay_score
        FREQUENCY:  score = access_count × trust_score
        HYBRID:     score = (iteration×0.4 + access×0.2 + bm25×0.2 + semantic×0.2) × trust × decay
        SEMANTIC:   score = cosine_similarity(query_emb, item_emb) × trust_score
        """
        policy_val = self.config.retrieval_policy.value
        if policy_val == "recent":
            return self.score_item(item, current_time)
        elif policy_val == "frequency":
            return item.access_count * item.trust
        elif policy_val == "hybrid":
            if not query:
                # Fallback without query: use RECENT-like scoring
                return self.score_item(item, current_time)
            age_minutes = (current_time - item.created_at) / 60.0
            half_life = self.config.decay_halflife_minutes
            decay_score = 0.5 ** (age_minutes / half_life)
            bm25_score = self._bm25_cache.get(item.id, 0.0)
            semantic_score = self._semantic_score(item, query) if item.id in self._item_embeddings else 0.0
            hybrid_signal = item.iteration * 0.4 + item.access_count * 0.2 + bm25_score * 0.2 + semantic_score * 0.2
            return hybrid_signal * item.trust * decay_score
        elif policy_val == "semantic":
            if not query:
                self._logger.warning("[EpisodicMemoryStore] SEMANTIC retrieval requires query, falling back to RECENT")
                return self.score_item(item, current_time)
            return self._semantic_score(item, query) * item.trust
        return 0.0

    def _load_embedding_model(self) -> bool:
        """Lazy-load the embedding model (sentence-transformers).

        Returns:
            True if model is available, False if not installed.
        """
        if EpisodicMemoryStore._embedding_model is not None:
            return True
        if EpisodicMemoryStore._embedding_available is False:
            return False
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            EpisodicMemoryStore._embedding_available = False
            self._logger.warning("[EpisodicMemoryStore] sentence-transformers not installed, semantic retrieval disabled")
            return False
        model_name = "all-MiniLM-L6-v2"
        self._logger.info(f"[EpisodicMemoryStore] Loading embedding model: {model_name}")
        EpisodicMemoryStore._embedding_model = SentenceTransformer(model_name)
        EpisodicMemoryStore._embedding_available = True
        self._logger.info("[EpisodicMemoryStore] Embedding model loaded")
        return True

    def _compute_item_embedding(self, item: MemoryItem) -> None:
        """Compute and cache embedding for a memory item."""
        if not self._load_embedding_model():
            return  # Model not available, skip embedding
        text = item.content[:512]
        if text in self._embedding_cache:
            embedding = self._embedding_cache[text]
        else:
            embedding = EpisodicMemoryStore._embedding_model.encode([text], normalize_embeddings=True)[0].tolist()
            self._embedding_cache[text] = embedding
        self._item_embeddings[item.id] = embedding

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for query with caching. Returns None if model unavailable."""
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        if not self._load_embedding_model():
            return None
        embedding = EpisodicMemoryStore._embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        self._embedding_cache[query] = embedding
        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _semantic_score(self, item: MemoryItem, query: str) -> float:
        """Calculate semantic similarity score between query and item content."""
        if item.id not in self._item_embeddings:
            self._compute_item_embedding(item)
        item_emb = self._item_embeddings.get(item.id)
        if not item_emb:
            return 0.0
        query_emb = self._get_query_embedding(query)
        if not query_emb:
            return 0.0
        return self._cosine_similarity(query_emb, item_emb)

    def _load_bm25_model(self) -> None:
        """Lazy-load the BM25 model."""
        if EpisodicMemoryStore._bm25_model is not None:
            return
        try:
            from rank_bm25 import BM25Plus
        except ImportError:
            self._logger.error("[EpisodicMemoryStore] rank-bm25 not installed. Install with: pip install rank-bm25")
            raise
        EpisodicMemoryStore._bm25_model = BM25Plus
        self._logger.info("[EpisodicMemoryStore] BM25Plus model loaded")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25 (lowercased, alphanumeric only)."""
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _compute_bm25_scores(self, query: str, items: List[MemoryItem]) -> None:
        """Pre-compute BM25 scores for all items against the query."""
        self._bm25_cache.clear()
        if not items:
            return

        try:
            from rank_bm25 import BM25Plus
        except ImportError:
            self._logger.warning("[EpisodicMemoryStore] rank-bm25 not available, skipping BM25 scores")
            return

        # Tokenize all item contents
        tokenized_docs = [self._tokenize_for_bm25(item.content) for item in items]
        if not any(tokenized_docs):
            return

        # Build BM25 index
        bm25 = BM25Plus(tokenized_docs)

        # Score query against all docs
        query_tokens = self._tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)

        # Cache scores by item id (normalize to 0-1 range)
        max_score = max(scores) if max(scores) > 0 else 1.0
        for item, score in zip(items, scores):
            self._bm25_cache[item.id] = score / max_score  # normalize to 0-1

    def _sort_by_policy(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """Sort items by retrieval policy."""
        if self.config.retrieval_policy == RetrievalPolicy.RECENT:
            # Most recent first (higher iteration, then later in list)
            return sorted(items, key=lambda x: (x.iteration, x.created_at), reverse=True)
        elif self.config.retrieval_policy == RetrievalPolicy.FREQUENCY:
            # Most accessed first
            return sorted(items, key=lambda x: x.access_count, reverse=True)
        elif self.config.retrieval_policy == RetrievalPolicy.HYBRID:
            # Hybrid: recent + frequency boost
            return sorted(
                items,
                key=lambda x: (x.iteration * 0.7 + x.access_count * 0.3, x.created_at),
                reverse=True,
            )
        return items

    def score_item(self, item: MemoryItem, current_time: float) -> float:
        """Calculate retrieval score for an item.

        Score = recency_score × trust_score × decay_score
        - recency_score = 1 / (1 + age_minutes / half_life)  # linear decay
        - trust_score = item.trust (0.0 ~ 1.0)
        - decay_score = 0.5^(age_minutes / half_life)  # exponential decay
        """
        age_minutes = (current_time - item.created_at) / 60.0
        half_life = self.config.decay_halflife_minutes

        # Linear recency score (0 to 1)
        recency_score = 1.0 / (1.0 + age_minutes / half_life)

        # Trust score (0.0 to 1.0)
        trust_score = item.trust

        # Exponential decay score (0 to 1)
        decay_score = 0.5 ** (age_minutes / half_life)

        return recency_score * trust_score * decay_score

    def update_trust(self, item_id: str, is_positive: bool) -> bool:
        """Update trust score for an item.

        Args:
            item_id: ID of the item to update
            is_positive: True for positive feedback (+0.05), False for negative (-0.10)

        Returns:
            True if item was found and updated, False otherwise
        """
        for item in self._items:
            if item.id == item_id:
                if is_positive:
                    item.trust = min(1.0, item.trust + 0.05)
                else:
                    item.trust = max(0.0, item.trust - 0.10)
                self._logger.debug(
                    f"[EpisodicMemoryStore] Updated trust for {item_id}: "
                    f"{'positive' if is_positive else 'negative'} -> {item.trust:.3f}"
                )
                return True
        return False

    def format_for_prompt(self, items: Optional[List[MemoryItem]] = None) -> str:
        """Format memory items as a string for injection into prompt.

        Args:
            items: Items to format. If None, retrieves based on config.

        Returns:
            Formatted memory context string
        """
        if items is None:
            items = self.retrieve()

        if not items:
            return ""

        lines = ["[CONTEXT FROM PREVIOUS STEPS - Memory]"]
        for i, item in enumerate(items, 1):
            source_tag = f"[{item.source}]"
            detail = f" ({item.source_detail})" if item.source_detail else ""
            # Truncate long content
            content = item.content
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{i}. {source_tag}{detail} -> \"{content}\"")
        lines.append("[END Memory]")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get memory summary for logging/transcript."""
        return {
            "total_items": len(self._items),
            "current_iteration": self._iteration,
            "enabled": self._enabled,
            "write_policy": self.config.write_policy.value,
            "retrieval_policy": self.config.retrieval_policy.value,
        }

    def should_write_event(self, event_type: str, content: str = "") -> bool:
        """Check if an event should be written based on write policy.

        Args:
            event_type: Type of event
            content: Content to potentially write

        Returns:
            True if should write
        """
        if not self._enabled:
            return False
        return should_write_to_memory(
            self.config.write_policy,
            event_type,
            content,
            self.config.long_content_threshold,
        )

    @property
    def item_count(self) -> int:
        """Current number of items in memory."""
        return len(self._items)

    @property
    def is_enabled(self) -> bool:
        """Whether memory is enabled."""
        return self._enabled

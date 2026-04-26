"""
Dense Passage Retriever - Recipe T4a: Program Support Cards

Uses a frozen BERT-based bi-encoder to compute cosine similarity
between task description embeddings and card embeddings.

Retrieval: frozen BERT bi-encoder → cosine similarity → top-k cards.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import ProgramSupportConfig, RetrievalConfig, SkillCard
from .event import ProceduralEvent

logger = logging.getLogger("agent.procedure.retriever")


class DenseRetriever:
    """Dense passage retriever using a frozen BERT bi-encoder.

    Computes embeddings for task descriptions and program support cards,
    then retrieves the top-k most similar cards via cosine similarity.
    """

    def __init__(self, config: ProgramSupportConfig):
        self.config = config
        self.retrieval_cfg = config.retrieval
        self._model = None
        self._tokenizer = None
        self._card_embeddings: Optional[np.ndarray] = None
        self._card_texts: List[str] = []
        self._card_names: List[str] = []
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "[DenseRetriever] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

        logger.info(
            f"[DenseRetriever] Loading embedding model: {self.retrieval_cfg.embedding_model}"
        )
        self._model = SentenceTransformer(
            self.retrieval_cfg.embedding_model,
            device=self.retrieval_cfg.device,
        )
        logger.info(f"[DenseRetriever] Model loaded on device: {self.retrieval_cfg.device}")

    def _build_card_text(self, card: SkillCard) -> str:
        """Build a dense text representation of a skill card for embedding.

        Uses only description + name prefix for clean semantic signal.
        Steps are intentionally excluded to avoid template noise (Step 1/2/3...)
        that dilutes description-level semantic matching.
        """
        parts = []
        if card.name:
            parts.append(f"Name: {card.name}")
        if card.description:
            parts.append(card.description)
        return " [SEP] ".join(parts)

    def index_cards(self, cards: List[SkillCard]) -> None:
        """Pre-compute embeddings for all program support cards.

        Args:
            cards: List of SkillCard objects to index
        """
        self._load_model()

        self._card_texts = []
        self._card_names = []

        for card in cards:
            text = self._build_card_text(card)
            self._card_texts.append(text)
            self._card_names.append(card.name)

        logger.info(f"[DenseRetriever] Encoding {len(cards)} cards...")
        self._card_embeddings = self._model.encode(
            self._card_texts,
            batch_size=self.retrieval_cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )
        logger.info(f"[DenseRetriever] Indexed {len(cards)} cards, embedding shape: {self._card_embeddings.shape}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[SkillCard, float, List[str]]]:
        """Retrieve top-k most relevant cards for a query.

        Args:
            query: Task description or context to match against cards
            top_k: Number of cards to retrieve (default from config)

        Returns:
            List of (SkillCard, similarity_score, matched_keywords) tuples
        """
        if self._card_embeddings is None or self._card_embeddings.size == 0:
            logger.warning("[DenseRetriever] No card embeddings indexed. Returning empty.")
            return []

        if top_k is None:
            top_k = self.retrieval_cfg.top_k

        # Encode query
        if query in self._embedding_cache and self.retrieval_cfg.cache_embeddings:
            query_embedding = self._embedding_cache[query]
        else:
            query_embedding = self._model.encode(
                [query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            if self.retrieval_cfg.cache_embeddings:
                self._embedding_cache[query] = query_embedding

        # Cosine similarity (embeddings are L2-normalized, so dot product = cosine)
        scores = np.dot(self._card_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[SkillCard, float, List[str]]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            # matched_keywords for backward compat: store top keyword match
            matched = [self._card_names[idx]]
            results.append((self._card_names[idx], score, matched))

        return results

    def get_card_count(self) -> int:
        """Get number of indexed cards."""
        return len(self._card_texts)

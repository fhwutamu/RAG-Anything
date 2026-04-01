from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "repos" / "lightrag"))
sys.path.insert(0, str(REPO_ROOT))

from raganything.modalprocessors import ImageModalProcessor


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class DummyKVStorage:
    def __init__(self, initial: dict[str, dict[str, Any]] | None = None) -> None:
        self.data = dict(initial or {})
        self.upserts: list[dict[str, dict[str, Any]]] = []

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    async def upsert(self, payload: dict[str, dict[str, Any]]) -> None:
        self.upserts.append(payload)
        self.data.update(payload)


class DummyGraph:
    def __init__(self) -> None:
        self.upsert_node = AsyncMock()
        self.upsert_edge = AsyncMock()


@dataclass
class DummyLightRAG:
    text_chunks: Any
    chunks_vdb: Any
    entities_vdb: Any
    relationships_vdb: Any
    chunk_entity_relation_graph: Any
    embedding_func: Any = None
    llm_model_func: Any = None
    multimodal_entity_extract_func: Any = None
    llm_response_cache: Any = None
    tokenizer: Any = field(default_factory=DummyTokenizer)

    async def _insert_done(self) -> None:
        return None


def _make_processor(initial_chunk: dict[str, Any] | None = None) -> ImageModalProcessor:
    text_chunks = DummyKVStorage({"chunk-001": initial_chunk} if initial_chunk else {})
    chunks_vdb = DummyKVStorage()
    entities_vdb = DummyKVStorage()
    relationships_vdb = DummyKVStorage()
    graph = DummyGraph()
    lightrag = DummyLightRAG(
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        chunk_entity_relation_graph=graph,
        multimodal_entity_extract_func=AsyncMock(),
    )
    return ImageModalProcessor(lightrag=lightrag, modal_caption_func=AsyncMock())


@pytest.mark.asyncio
async def test_process_chunk_for_extraction_prefers_multimodal_extractor():
    processor = _make_processor(
        initial_chunk={
            "tokens": 12,
            "content": "Figure 1 raw chunk",
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
            "file_path": "paper.pdf",
            "multimodal_payload": {
                "item_type": "image",
                "image": "/tmp/figure.png",
                "text": "Figure 1 metadata",
            },
        }
    )

    multimodal_result = [
        (
            {
                "Axis Label": [
                    {
                        "entity_name": "Axis Label",
                        "entity_type": "Data",
                        "description": "A label from the figure.",
                        "source_id": "chunk-001",
                        "file_path": "paper.pdf",
                        "timestamp": 1,
                    }
                ]
            },
            {},
        )
    ]

    with (
        patch(
            "raganything.modalprocessors.extract_multimodal_entities",
            new=AsyncMock(return_value=multimodal_result),
        ) as multimodal_extract,
        patch(
            "raganything.modalprocessors.extract_entities",
            new=AsyncMock(return_value=[]),
        ) as text_extract,
        patch(
            "raganything.modalprocessors.get_namespace_data",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "raganything.modalprocessors.get_pipeline_status_lock",
            return_value=asyncio.Lock(),
        ),
    ):
        result = await processor._process_chunk_for_extraction(
            "chunk-001", "Figure 1", batch_mode=True
        )

    multimodal_extract.assert_awaited_once()
    text_extract.assert_not_awaited()
    assert ("Axis Label", "Figure 1") in result[0][1]


@pytest.mark.asyncio
async def test_create_entity_and_chunk_passes_embedding_payload_to_vector_store():
    processor = _make_processor()
    entity_info = {
        "entity_name": "Figure 1",
        "entity_type": "image",
        "summary": "A chart figure.",
    }

    with patch.object(
        processor,
        "_process_chunk_for_extraction",
        new=AsyncMock(return_value=[]),
    ):
        await processor._create_entity_and_chunk(
            modal_chunk="Figure 1 raw chunk",
            entity_info=entity_info,
            file_path="paper.pdf",
            doc_id="doc-001",
            embedding_payload={"image": "/tmp/figure.png", "text": "Figure 1 metadata"},
            multimodal_payload={"item_type": "image", "image": "/tmp/figure.png"},
            context_text="Nearby paragraph text",
        )

    chunk_payload = processor.chunks_vdb.upserts[0]
    stored_chunk = next(iter(chunk_payload.values()))
    assert stored_chunk["content"] == "Figure 1 raw chunk"
    assert stored_chunk["embedding_content"] == {
        "image": "/tmp/figure.png",
        "text": "Figure 1 metadata",
    }

    stored_text_chunk = processor.text_chunks_db.data[next(iter(processor.text_chunks_db.data))]
    assert stored_text_chunk["multimodal_payload"] == {
        "item_type": "image",
        "image": "/tmp/figure.png",
    }
    assert stored_text_chunk["context_text"] == "Nearby paragraph text"

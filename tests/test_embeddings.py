from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "repos" / "lightrag"))
sys.path.insert(0, str(REPO_ROOT))

from raganything.embeddings import (
    create_qwen3_vl_embedding_func,
    normalize_multimodal_embedding_item,
)


def test_normalize_multimodal_embedding_item_uses_image_and_text_fields():
    normalized = normalize_multimodal_embedding_item(
        {
            "item_type": "image",
            "text": "Figure caption",
            "image": "/tmp/figure.png",
        }
    )

    assert normalized == {
        "text": "Figure caption",
        "image": "/tmp/figure.png",
    }


@pytest.mark.asyncio
async def test_create_qwen3_vl_embedding_func_normalizes_batch_inputs():
    seen: list[list[dict[str, str]]] = []

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def embed(self, items):
            seen.append(list(items))
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    embedding_func = create_qwen3_vl_embedding_func(
        "Qwen/Qwen3-VL-Embedding-8B",
        embedding_dim=2,
        adapter_cls=FakeAdapter,
    )

    embeddings = await embedding_func(
        [
            {"text": "caption", "image": "/tmp/figure.png"},
            "plain text query",
        ]
    )

    assert embeddings.shape == (2, 2)
    assert seen == [
        [
            {"text": "caption", "image": "/tmp/figure.png"},
            {"text": "plain text query"},
        ]
    ]

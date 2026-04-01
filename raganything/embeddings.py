from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from lightrag.utils import EmbeddingFunc

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_FRAMES = 64
FPS = 1
MAX_TOTAL_PIXELS = 10 * (768 * IMAGE_FACTOR * IMAGE_FACTOR)


def normalize_multimodal_embedding_item(item: Any) -> dict[str, Any]:
    """Normalize arbitrary batch items into the Qwen3-VL text/image input shape."""

    if isinstance(item, str):
        return {"text": item}

    if not isinstance(item, dict):
        return {"text": str(item)}

    content = item.get("content")
    normalized: dict[str, Any] = {}

    text = item.get("text")
    if text is None and isinstance(content, dict):
        text = (
            content.get("text")
            or content.get("table_body")
            or content.get("equation")
            or content.get("content")
        )
    if text is None and "content" in item and not isinstance(content, dict):
        text = item["content"]

    image = item.get("image") or item.get("img_path")
    if image is None and isinstance(content, dict):
        image = content.get("image") or content.get("img_path")

    instruction = item.get("instruction")
    video = item.get("video")

    if image:
        normalized["image"] = str(image)
    if text is not None and str(text).strip():
        normalized["text"] = str(text)
    if instruction:
        normalized["instruction"] = str(instruction)
    if video:
        normalized["video"] = video

    if not normalized:
        normalized["text"] = "NULL"
    return normalized


def _build_qwen3_vl_model_class():
    import torch
    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    @dataclass
    class _Qwen3VLEmbeddingOutput(ModelOutput):
        last_hidden_state: torch.FloatTensor | None = None
        attention_mask: torch.Tensor | None = None

    class _Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            cache_position=None,
            **kwargs,
        ):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )
            return _Qwen3VLEmbeddingOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

    return _Qwen3VLForEmbedding


class Qwen3VLEmbeddingAdapter:
    """In-process Qwen3-VL embedding runtime for text+image payloads."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        device: str | None = None,
        torch_dtype: Any | None = None,
        **model_kwargs,
    ) -> None:
        import torch
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self._torch = torch
        self._F = torch.nn.functional
        self._process_vision_info = __import__(
            "qwen_vl_utils.vision_process",
            fromlist=["process_vision_info"],
        ).process_vision_info

        model_cls = _build_qwen3_vl_model_class()
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = model_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        ).to(resolved_device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )
        self.model.eval()

    def format_model_input(
        self,
        *,
        text: str | None = None,
        image: str | Any | None = None,
        video: Any | None = None,
        instruction: str | None = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction or self.default_instruction,
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

        if image is not None:
            image_value = image
            if isinstance(image_value, str) and not image_value.startswith(("http", "oss", "file://")):
                image_value = "file://" + image_value
            content.append(
                {
                    "type": "image",
                    "image": image_value,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        if video is not None:
            content.append(
                {
                    "type": "video",
                    "video": video,
                    "fps": self.fps,
                    "max_frames": self.max_frames,
                    "total_pixels": self.total_pixels,
                }
            )

        content.append({"type": "text", "text": text or "NULL"})
        return conversation

    def _preprocess_inputs(self, conversations: list[list[dict[str, Any]]]):
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = self._process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos = None
            video_metadata = None

        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pool_last_token(hidden_state, attention_mask):
        flipped_mask = attention_mask.flip(dims=[1])
        last_token_positions = flipped_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_token_positions - 1
        import torch

        row_tensor = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row_tensor, col]

    def embed(self, items: Sequence[dict[str, Any]], normalize: bool = True) -> np.ndarray:
        conversations = [
            self.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
            )
            for item in items
        ]
        model_inputs = self._preprocess_inputs(conversations)
        model_inputs = {
            key: value.to(self.model.device) for key, value in model_inputs.items()
        }
        with self._torch.no_grad():
            outputs = self.model(**model_inputs)
            embeddings = self._pool_last_token(
                outputs.last_hidden_state,
                outputs.attention_mask,
            )
            if normalize:
                embeddings = self._F.normalize(embeddings, p=2, dim=-1)
        return embeddings.detach().cpu().numpy()


def create_qwen3_vl_embedding_func(
    model_name_or_path: str,
    *,
    embedding_dim: int = 4096,
    max_length: int = MAX_LENGTH,
    default_instruction: str = "Represent the user's input.",
    adapter_cls=Qwen3VLEmbeddingAdapter,
    **adapter_kwargs,
) -> EmbeddingFunc:
    """Create a LightRAG-compatible EmbeddingFunc backed by Qwen3-VL-Embedding."""

    adapter_holder: dict[str, Any] = {}

    async def _embed(batch: Sequence[Any], **kwargs) -> np.ndarray:
        if "adapter" not in adapter_holder:
            adapter_holder["adapter"] = adapter_cls(
                model_name_or_path,
                max_length=max_length,
                default_instruction=default_instruction,
                **adapter_kwargs,
            )
        adapter = adapter_holder["adapter"]
        normalized_batch = [
            normalize_multimodal_embedding_item(item) for item in batch
        ]
        return adapter.embed(normalized_batch)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=_embed,
        max_token_size=max_length,
        model_name=model_name_or_path,
    )

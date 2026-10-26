"""Image-only processor over the existing DeepSeek native chat renderer."""

import json
from copy import deepcopy
from pathlib import Path

from miles_plugins.models.deepseek_v4_1.image_processing import expand_image_tokens
from miles_plugins.models.deepseek_v4_1.vision import VisionConfig


class DeepSeekV41Processor:
    def __init__(self, hf_checkpoint):
        from miles.utils.processing_utils import load_tokenizer

        config = json.loads((Path(hf_checkpoint) / "config.json").read_text())
        if not config.get("vision_config"):
            raise ValueError("DeepSeek image SFT requires the original multimodal config")
        self.config = VisionConfig.from_dict(config["vision_config"])
        self.image_token_id = config["image_token_id"]
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.image_token = self.tokenizer.convert_ids_to_tokens(self.image_token_id)

    def extract_media(self, messages):
        images = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                continue
            for part in content:
                if part["type"] == "text":
                    continue
                if part["type"] != "image":
                    raise ValueError("V4.1 training accepts text and resolved image parts only")
                images.append(part["image"])
        return {"images": images}

    def normalize_messages(self, messages, images=None):
        normalized = deepcopy(messages)
        image_iter = iter(images or [])
        for message in normalized:
            content = message.get("content", "")
            if isinstance(content, str):
                if "<image>" not in content:
                    continue
                parts = content.split("<image>")
                content = []
                for index, text in enumerate(parts):
                    if index:
                        try:
                            image = next(image_iter)
                        except StopIteration as error:
                            raise ValueError("Image placeholders require resolved image inputs") from error
                        content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": text})
            blocks = []
            for part in content:
                if part["type"] == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    blocks.append({"type": "image_url", "image_url": {"url": str(part["image"])}})
                else:
                    raise ValueError("V4.1 training does not support video/audio parts")
            if all(block["type"] == "text" for block in blocks):
                message["content"] = "".join(block["text"] for block in blocks)
            else:
                # The native encoder rejects hand-written image special tokens.
                message.pop("content", None)
                message["content_blocks"] = blocks
        return normalized

    def __call__(self, text, images=None, **kwargs):
        from miles.utils.chat_template_utils.deepseek import apply_chat_template

        if any(kwargs.get(key) for key in ("videos", "audios", "audio")):
            raise ValueError("DeepSeek V4.1 processor supports images only")
        if isinstance(text, list):
            text = apply_chat_template(
                self.normalize_messages(text, images), self.tokenizer, add_generation_prompt=True
            )
        else:
            text = text.replace("<image>", self.image_token)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        expanded, _, inputs = expand_image_tokens(tokens, images or [], self.image_token_id, self.config)
        return {"input_ids": [expanded], **(inputs or {})}

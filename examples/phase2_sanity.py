#!/usr/bin/env python3
"""Phase-2 sanity: prove SmolVLM2-2.2B actually perceives a REAL photo.

Loads the bigger model and describes examples/images/cats.jpg (two tabby cats
on a pink blanket with two TV remotes). A pass = the description names the cats
(and ideally the remotes), proving this model does real object perception, not
just the color-only behaviour of the 500M tracer-bullet model.
"""

from __future__ import annotations

import sys

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "examples/images/cats.jpg"


def main() -> None:
    print(f"[load] {MODEL}")
    model, processor = load(MODEL)
    print("[loaded] describing", IMAGE)

    question = "Describe this image in one sentence."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    prompt = apply_chat_template(processor, model.config, messages, num_images=1)
    out = generate(
        model,
        processor,
        prompt,
        image=[IMAGE],
        max_tokens=80,
        temperature=0.0,
        verbose=False,
    )
    text = out.text if hasattr(out, "text") else str(out)
    print("\nANSWER:", repr(text.strip()))

    low = text.lower()
    has_cat = "cat" in low
    print("\nmentions 'cat':", has_cat)
    print("SANITY:", "PASS" if has_cat else "CHECK")


if __name__ == "__main__":
    main()

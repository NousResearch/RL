"""Convert lambda/hermes-agent-reasoning-traces to NeMo RL JSONL format.

Initially generated on 2026-04-03 to convert the lambda/hermes-agent-reasoning-traces dataset to a usable format for SFT...

The dataset uses ShareGPT/Hermes conventions: conversations are stored under
a `conversations` key with `from`/`value` turn dicts, and role names differ
from the OpenAI standard (human -> user, gpt -> assistant).

Tool definitions are already embedded verbatim in the system message as
<tools>...</tools> XML, so use tokenizer.chat_template: NULL (passthrough)
when training — the content is already formatted for the Hermes chat template.

Usage:
  # default dataset
  uv run scripts/convert_ds_to_nemorl_format.py --output_dir /path/to/output

  # custom dataset
  uv run scripts/convert_ds_to_nemorl_format.py --dataset org/my-dataset --output_dir /path/to/output

  
Suggested NeMo RL YAML snippet after running:
    tokenizer:
      chat_template: NULL

    data:
      train:
        dataset_name: openai_format
        data_path: /path/to/output/train.jsonl
        chat_key: messages
        tool_key: tools
        use_preserving_dataset: true
      validation:
        dataset_name: openai_format
        data_path: /path/to/output/val.jsonl
        chat_key: messages
        tool_key: tools
        use_preserving_dataset: true
"""

import argparse
import json
import os
import random

from datasets import load_dataset

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}


def convert_sample(sample: dict) -> dict:
    messages = []
    for turn in sample["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            raise ValueError(f"Unexpected role '{turn['from']}' in sample id={sample.get('id')}")
        messages.append({"role": role, "content": turn["value"]})

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Last turn must be from the assistant, got '{messages[-1]['role']}' "
            f"in sample id={sample.get('id')}"
        )

    result: dict = {"messages": messages}

    # The `tools` column is a JSON string; parse it into a list.
    # With passthrough template this isn't used for formatting, but including
    # it preserves the data for users who switch to a structured template later.
    tools_raw = sample.get("tools")
    if tools_raw:
        try:
            result["tools"] = json.loads(tools_raw)
        except (json.JSONDecodeError, TypeError):
            pass  # drop malformed tools rather than corrupting the sample

    return result


def write_jsonl(path: str, samples: list[dict]) -> None:
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote {len(samples):,} samples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ShareGPT/Hermes-format HuggingFace dataset to NeMo RL JSONL format."
    )
    parser.add_argument(
        "--dataset",
        default="lambda/hermes-agent-reasoning-traces",
        help="HuggingFace dataset name (default: lambda/hermes-agent-reasoning-traces).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where train.jsonl (and optionally val.jsonl) will be written.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation (default: 0.05). Pass 0 to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val shuffle (default: 42).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset} from HuggingFace...")
    ds = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(ds):,} samples.")

    samples = []
    n_skipped = 0
    for raw in ds:
        try:
            samples.append(convert_sample(raw))
        except ValueError as e:
            print(f"  Skipping sample: {e}")
            n_skipped += 1

    if n_skipped:
        print(f"Skipped {n_skipped} malformed samples.")

    if args.val_split > 0:
        random.seed(args.seed)
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * args.val_split))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]
    else:
        train_samples = samples
        val_samples = []

    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
    if val_samples:
        write_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)

    print("\nDone. Add this to your sft.yaml (adjust paths as needed):")
    train_path = os.path.abspath(os.path.join(args.output_dir, "train.jsonl"))
    val_path = os.path.abspath(os.path.join(args.output_dir, "val.jsonl"))
    print(f"""
  tokenizer:
    chat_template: NULL   # passthrough — content is already Hermes-formatted

  data:
    train:
      dataset_name: openai_format
      data_path: {train_path}
      chat_key: messages
      tool_key: tools
      use_preserving_dataset: true   # tools have heterogeneous argument schemas
    validation:
      dataset_name: openai_format
      data_path: {val_path}
      chat_key: messages
      tool_key: tools
      use_preserving_dataset: true
""")


if __name__ == "__main__":
    main()

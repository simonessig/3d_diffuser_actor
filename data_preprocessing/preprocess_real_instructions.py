"""
Precompute embeddings of instructions.
"""

import itertools
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import tap
import torch
import transformers
from tqdm.auto import tqdm

TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    output: Path = "instructions/real/pick_box/instructions.pkl"
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    device: str = "cuda"
    verbose: bool = False
    annotation_path: Path = "data/real/raw/pick_box/annotations.json"


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


def main(args):
    with open(str(args.annotation_path), "r") as j:
        annotations = json.loads(j.read())

    instructions_string = annotations["ann"]

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder)
    model = model.to(args.device)

    instructions = {"embeddings": [], "text": []}

    for instr in tqdm(instructions_string):
        tokens = tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(args.device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state
        instructions["embeddings"].append(pred.cpu())
        instructions["text"].append(instr)

    os.makedirs(str(args.output.parent), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(instructions, f)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    main(args)

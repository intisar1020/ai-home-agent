#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import torch
import tiktoken

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.gpt_model import GPTModel
from utils.generate import generate_with_cache

PROMPT_FMT = "### Query:\n{query}\n\n### Tool Call:\n"


def load_model(model_cfg, checkpoint_path):
    model = GPTModel(model_cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_tool_name(text, prompt):
    response = text[len(prompt) :] if text.startswith(prompt) else text
    m = re.search(r'"tool"\s*:\s*"(\w+)"', response)
    return m.group(1) if m else ""


def generate_one(model, tokenizer, query, max_new_tokens, temperature, top_k, top_p):
    prompt = PROMPT_FMT.format(query=query)
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with torch.no_grad():
        generated_ids = generate_with_cache(
            model=model,
            idx=input_tensor,
            max_new_tokens=max_new_tokens,
            context_size=1024,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
        )

    full_text = tokenizer.decode(generated_ids[0].tolist())
    tool_name = parse_tool_name(full_text, prompt)
    response = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text

    return response, tool_name


def interactive_loop(model, tokenizer, args):
    print("\nEnter queries (empty line or Ctrl+C to quit):\n")
    try:
        while True:
            query = input("Query> ").strip()
            if not query:
                break
            response, tool = generate_one(
                model, tokenizer, query,
                args.max_tokens, args.temperature, args.top_k, args.top_p,
            )
            print(f"Tool:  {tool or '(none)'}")
            print(f"Raw:   {response.strip()}")
            print(f"Latency: -- (no timing)\n")
    except (KeyboardInterrupt, EOFError):
        pass


def eval_dataset(model, tokenizer, args):
    data_path = Path(args.eval)
    if not data_path.exists():
        data_path = SCRIPT_DIR.parent / args.eval

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Evaluating on {len(samples)} samples ...\n")
    correct = 0

    for i, sample in enumerate(samples):
        query = sample["query"]
        expected_tool = sample["tool"]
        _, generated_tool = generate_one(
            model, tokenizer, query,
            args.max_tokens, args.temperature, args.top_k, args.top_p,
        )
        ok = generated_tool == expected_tool
        if ok:
            correct += 1

        if i < 10 or not ok:
            status = "OK" if ok else f"MISS (expected={expected_tool}, got={generated_tool})"
            print(f"  [{i:3d}] {status:50s}  | {query[:60]}...")

    acc = correct / len(samples) * 100
    print(f"\nAccuracy: {correct}/{len(samples)} ({acc:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="AI Monitor — Tool-calling inference")
    parser.add_argument("--ckpt", default=None, help="Path to .pth or .ckpt checkpoint")
    parser.add_argument("--config", default=None, help="Path to config JSON (default: configs/tools_gpt2.json)")
    parser.add_argument("--query", "-q", default=None, help="Single query to test")
    parser.add_argument("--interactive", "-i", action="store_true", default=False)
    parser.add_argument("--eval", default=None, help="Path to JSONL dataset for batch evaluation")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else SCRIPT_DIR / "configs" / "tools_gpt2.json"
    with open(config_path) as f:
        cfg = json.load(f)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        candidates = sorted(SCRIPT_DIR.glob("logs/checkpoints/ai_monitor*.pth"))
        if not candidates:
            print("No checkpoints found. Use --ckpt to specify one.")
            sys.exit(1)
        ckpt_path = str(candidates[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    model = load_model(cfg["model"], ckpt_path)
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Config: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    if args.eval:
        eval_dataset(model, tokenizer, args)
    elif args.query:
        response, tool = generate_one(
            model, tokenizer, args.query,
            args.max_tokens, args.temperature, args.top_k, args.top_p,
        )
        print(f"\nQuery:  {args.query}")
        print(f"Tool:   {tool or '(none)'}")
        print(f"Output: {response.strip()}")
    else:
        interactive_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()

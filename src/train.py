import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import tiktoken

from dataset.tools_datamodule import ToolsDataModule
from trainer_engine.lit_gpt import LitGPT
from trainer_engine.tool_validator import ToolCallAccuracyCallback
from pytorch_lightning.loggers import WandbLogger


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_jsonl(data_path):
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def main():
    config_path = SCRIPT_DIR / "configs" / "tools_gpt2.json"
    cfg = load_config(config_path)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_path_str = cfg["data"]["data_path"]
    data_path = PROJECT_ROOT / data_path_str if not os.path.isabs(data_path_str) else Path(data_path_str)

    print(f"Loading dataset from: {data_path}")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} samples")

    if len(data) == 0:
        raise ValueError("Dataset is empty. Check data_path in config.")

    random.shuffle(data)
    train_portion = int(0.90 * len(data))
    train_data = data[:train_portion]
    val_data = data[train_portion:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "ai_monitor"),
        name=cfg.get("wandb_name", "exp_1.1"),
        log_model="all",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="logs/checkpoints",
        filename="ai_monitor-{epoch:02d}-{val_loss:.4f}",
        save_top_k=2,
        mode="min",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cfg.get("early_stop_patience", 5),
        mode="min",
    )

    tool_eval_samples = val_data[:20]
    tool_accuracy_callback = ToolCallAccuracyCallback(
        eval_samples=tool_eval_samples,
        tokenizer=tokenizer,
        max_new_tokens=cfg.get("tool_eval_max_new_tokens", 128),
        every_n_epochs=cfg.get("tool_eval_every_n_epochs", 5),
    )

    data_module = ToolsDataModule(
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pad_token_id=pad_token_id,
        ignore_index=cfg["data"]["ignore_index"],
        max_length=cfg["data"]["max_length"],
        device=cfg["data"].get("device", "cpu"),
    )

    if cfg["train"].get("pretrained_path"):
        pt_path = cfg["train"]["pretrained_path"]
        if not os.path.isabs(pt_path):
            cfg["train"]["pretrained_path"] = str(SCRIPT_DIR / pt_path)

    model = LitGPT(cfg["model"], cfg["train"])
    model.pad_token_id = pad_token_id
    model.tokenizer = tokenizer

    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=[checkpoint_callback, early_stop_callback, tool_accuracy_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, data_module)

    ckpt_dir = Path("logs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint(ckpt_dir / "ai_monitor_final.ckpt")
    torch.save(model.state_dict(), ckpt_dir / "ai_monitor_final.pth")
    print(f"Training completed. Checkpoints saved to {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()

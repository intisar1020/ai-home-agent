# instruct_datamodule_next_token.py
# Simple next-token prediction version

import json

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from typing import List, Optional


class ToolsDataset(Dataset):
    """
    Dataset for tool-calling next-token prediction.

    Each sample encodes a user query as input and a JSON dict
    containing the selected tool and its arguments as output.
    The model learns to generate the tool-call dict token by token.

    Example JSON entry:
    {
        "query": "Was there any motion detected on the driveway camera between 6 PM and 9 PM yesterday evening?",
        "tool": "detect_motion",
        "arguments": {"camera": "driveway"}
    }

    Encoded into training text:
    ### Query:
    <query text>

    ### Tool Call:
    {"tool": "detect_motion", "arguments": {"camera": "driveway"}}
    """

    def __init__(self, json_data: List[dict], tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_text: List[List[int]] = []

        for entry in json_data:
            query = entry.get("query", "")
            tool = entry.get("tool", "")
            arguments = entry.get("arguments", {})

            tool_call_dict = {"tool": tool, "arguments": arguments}
            tool_call_str = json.dumps(tool_call_dict)

            text = f"### Query:\n{query}\n\n### Tool Call:\n{tool_call_str}"

            ids = self.tokenizer.encode(text)
            self.encoded_text.append(ids)

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, idx):
        return self.encoded_text[idx]


def custom_collate_fn(
    batch,
    pad_token_id: int = 50256,
    device: str = "cpu",
    allowed_max_length: Optional[int] = None,
    ignore_index: int = -100
):
    #  max length of batch
    batch_max_length = max(len(item) + 1 for item in batch)  # +1 for added pad_token
    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()
        # an <|endoftext|> token
        new_item += [pad_token_id]
        # pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # shift the inputs to the right
        targets = torch.tensor(padded[1:])  # shift the targets to the left

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor



class ToolsDataModule(pl.LightningDataModule):
    """Lightning DataModule for next-token prediction."""

    def __init__(
        self,
        tokenizer,
        train_data: List[dict],
        val_data: Optional[List[dict]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.device = device

    def setup(self, stage=None):
        self.train_dataset = ToolsDataset(
            self.train_data, self.tokenizer, max_length=self.max_length
        )
        self.val_dataset = (
            ToolsDataset(self.val_data, self.tokenizer, max_length=self.max_length)
            if self.val_data
            else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                custom_collate_fn,
                pad_token_id=self.pad_token_id,
                device=self.device,
                ignore_index=self.ignore_index,
                allowed_max_length=self.max_length,
            ),
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(
                custom_collate_fn,
                pad_token_id=self.pad_token_id,
                device=self.device,
                ignore_index=self.ignore_index,
                allowed_max_length=self.max_length,
            ),
        )

if __name__ == "__main__":
    import tiktoken

    sample_data = [
        {
            "query": "Was there any motion detected on the driveway camera between 6 PM and 9 PM yesterday evening?",
            "tool": "detect_motion",
            "arguments": {"camera": "driveway"}
        },
        {
            "query": "Is the front door locked right now?",
            "tool": "check_lock",
            "arguments": {"door": "front"}
        },
        {
            "query": "Turn off all the lights in the living room.",
            "tool": "control_lights",
            "arguments": {"room": "living_room", "action": "off"}
        },
    ]

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = ToolsDataset(sample_data, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(custom_collate_fn, pad_token_id=50256)
    )

    print("=== Decoded text examples ===\n")
    for i, ids in enumerate(dataset):
        print(f"--- Sample {i} ---")
        print(tokenizer.decode(ids))
        print()

    print("=== Dataloader batch (inputs, targets shapes) ===")
    for inputs, targets in dataloader:
        print(f"inputs shape:  {inputs.shape}")
        print(f"targets shape: {targets.shape}")
        print (f"Decoded output: {tokenizer.decode(inputs[0][:-5].cpu().numpy())}")
        print(f"inputs[0][:20]:  {inputs[0][:5].tolist()}")
        print(f"targets[0][:20]: {targets[0][:5].tolist()}")
        break
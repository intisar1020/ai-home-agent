import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.gpt_model import GPTModel


class LitGPT(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "train_cfg"])

        self.lr = train_cfg["lr"]
        self.weight_decay = train_cfg.get("weight_decay", 0.1)
        self.warmup_steps = train_cfg.get("warmup_steps", 500)
        self.max_epochs = train_cfg.get("epoch", 50)

        self.model = GPTModel(model_cfg)
        pretrained_path = train_cfg.get("pretrained_path", "")
        if pretrained_path:
            self.model.load_state_dict(
                torch.load(pretrained_path, map_location="cpu", weights_only=True)
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, use_cache=False):
        return self.model(x, use_cache=use_cache)

    def reset_kv_cache(self):
        self.model.reset_kv_cache()

    def _compute_loss(self, batch):
        input_batch, target_batch = batch
        logits = self(input_batch)

        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_batch.view(-1)

        ignore_index = self.criterion.ignore_index
        valid_mask = target_flat != ignore_index
        num_valid_tokens = valid_mask.sum().item()

        loss = self.criterion(logits_flat, target_flat)

        return loss, num_valid_tokens

    def training_step(self, batch, batch_idx):
        loss, num_valid_tokens = self._compute_loss(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_tokens", float(num_valid_tokens), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, num_valid_tokens = self._compute_loss(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_tokens", float(num_valid_tokens), prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_train_steps = (
            self.max_epochs
            * self.trainer.estimated_stepping_batches
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = (current_step - self.warmup_steps) / max(
                1, total_train_steps - self.warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

import re
import torch
import pytorch_lightning as pl
from utils.generate import generate_with_cache


def _find_tool_name(generated_text, prompt):
    response = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text

    m = re.search(r'"tool"\s*:\s*"(\w+)"', response)
    if m:
        return m.group(1)

    return ""


class ToolCallAccuracyCallback(pl.Callback):
    def __init__(self, eval_samples, tokenizer, max_new_tokens=128, every_n_epochs=5):
        super().__init__()
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        model = pl_module.model
        model.eval()

        correct = 0
        total = len(self.eval_samples)

        for sample in self.eval_samples:
            query = sample["query"]
            expected_tool = sample["tool"]

            prompt = f"### Query:\n{query}\n\n### Tool Call:\n"
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(pl_module.device)

            with torch.no_grad():
                generated_ids = generate_with_cache(
                    model=model,
                    idx=input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    context_size=1024,
                    temperature=0.0,
                    top_k=None,
                    top_p=None,
                    eos_id=None,
                )

            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            generated_tool = _find_tool_name(generated_text, prompt)

            if generated_tool in expected_tool:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        pl_module.log("tool_accuracy", accuracy, on_epoch=True, prog_bar=True)
        print(
            f"\n[Epoch {trainer.current_epoch:3d}] Tool Accuracy: "
            f"{accuracy:.2%} ({correct}/{total})"
        )

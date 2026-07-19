#credit: following code/function is adapted from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01-main-chapter-code/previous_chapters.py

import torch

def generate(model, idx, max_new_tokens, context_size, temperature=0.0,
             top_k=None, top_p=None, repetition_penalty=1.0, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            generated_ids = idx[0].tolist()
            for token_id in set(generated_ids):
                logits[:, token_id] /= repetition_penalty

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs - sorted_probs > top_p
                mask[:, 0] = False
                mask = mask.scatter(1, sorted_indices, mask)
                probs = probs.masked_fill(mask, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_with_cache(model, idx, max_new_tokens, context_size=1024, temperature=0.0,
                        top_k=40, top_p=None, repetition_penalty=1.0, eos_id=None):
    model.eval()
    with torch.no_grad():
        model.reset_kv_cache()
        logits = model(idx[:, -context_size:], use_cache=True)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                generated_ids = idx[0].tolist()
                for token_id in set(generated_ids):
                    logits[:, token_id] /= repetition_penalty

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            if temperature > 0.0:
                logits = logits / temperature
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1)

                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs - sorted_probs > top_p
                    mask[:, 0] = False
                    mask = mask.scatter(1, sorted_indices, mask)
                    probs = probs.masked_fill(mask, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

            logits = model(idx_next, use_cache=True)

    model.reset_kv_cache()
    return idx
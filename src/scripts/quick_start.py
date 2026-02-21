import argparse
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="LaSER quick embedding demo")
    parser.add_argument("--model_path", type=str, required=True, help="Merged checkpoint path or HF model id")
    parser.add_argument("--query", type=str, required=True, help="One query text")
    parser.add_argument("--doc", type=str, required=True, help="One document text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_thinking_steps", type=int, default=3)
    return parser.parse_args()


@torch.inference_mode()
def laser_encode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    num_thinking_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if num_thinking_steps < 1:
        raise ValueError("num_thinking_steps must be >= 1")

    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    batch_size = input_ids.size(0)
    thinking_slots = num_thinking_steps - 1
    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")

    if thinking_slots > 0:
        eos_padding = torch.full((batch_size, thinking_slots), eos_id, dtype=input_ids.dtype, device=device)
        mask_padding = torch.ones((batch_size, thinking_slots), dtype=attention_mask.dtype, device=device)
        input_ids = torch.cat([input_ids, eos_padding], dim=1)
        attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

    input_embeds = model.get_input_embeddings()(input_ids)
    embedding_table = model.get_input_embeddings().weight
    base_seq_len = input_embeds.size(1) - thinking_slots

    past_key_values = None
    hidden_steps = []

    for step_idx in range(thinking_slots):
        current_use_idx = base_seq_len + step_idx

        if past_key_values is None:
            step_embeds = input_embeds[:, :current_use_idx, :]
        else:
            step_embeds = input_embeds[:, current_use_idx - 1 : current_use_idx, :]

        step_mask = attention_mask[:, :current_use_idx]

        outputs = model(
            inputs_embeds=step_embeds,
            attention_mask=step_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        hidden_steps.append(outputs.hidden_states[-1][:, -1, :])

        token_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        new_embed = token_probs @ embedding_table
        past_key_values = outputs.past_key_values

        pre_embeds = input_embeds[:, :current_use_idx, :]
        post_embeds = input_embeds[:, current_use_idx + 1 :, :]
        input_embeds = torch.cat([pre_embeds, new_embed.unsqueeze(1), post_embeds], dim=1)

    if past_key_values is None:
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
    else:
        outputs = model(
            inputs_embeds=input_embeds[:, -1:, :],
            attention_mask=attention_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    hidden_steps.append(outputs.hidden_states[-1][:, -1, :])

    embeddings = torch.stack(hidden_steps, dim=1).mean(dim=1)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    query_emb = laser_encode(
        model,
        tokenizer,
        [args.query],
        max_length=args.max_length,
        num_thinking_steps=args.num_thinking_steps,
        device=device,
    )
    doc_emb = laser_encode(
        model,
        tokenizer,
        [args.doc],
        max_length=args.max_length,
        num_thinking_steps=args.num_thinking_steps,
        device=device,
    )

    sim = (query_emb @ doc_emb.T).item()

    print(f"query embedding shape: {tuple(query_emb.shape)}")
    print(f"doc embedding shape: {tuple(doc_emb.shape)}")
    print(f"cosine similarity: {sim:.6f}")


if __name__ == "__main__":
    main()
# <div align="center"> LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval </div>

<div align="center">


<a href="https://arxiv.org/abs/2603.01425" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-b5212f.svg?logo=arxiv"></a>
<a href="https://huggingface.co/Alibaba-NLP/LaSER-Qwen3-0.6B" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-LaSER-blue"></a>
<a href="https://github.com/ignorejjj/LaSER"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Python" src="https://img.shields.io/badge/made_with-Python-blue"></a>

</div>

---

## 📖 Introduction

**LaSER** is a novel self-distillation framework designed to bridge the gap between powerful but slow explicit reasoning (like Chain-of-Thought) and efficient but shallow dense retrieval. While Large Language Models (LLMs) possess strong reasoning capabilities, current retrievers often treat them as static encoders, leaving their reasoning potential dormant.

![Introduction](<./assets/intro.png>)

Existing "rewrite-then-retrieve" pipelines suffer from prohibitive latency due to autoregressive text generation. In contrast, LaSER internalizes explicit reasoning into the **latent space**, allowing the retriever to "think" silently through continuous latent tokens. This enables the model to successfully combine the reasoning depth of explicit CoT pipelines with the inference efficiency of standard dense retrievers.

## ✨ Key Features

- 🧠 **Latent Thinking Mechanism**: Replaces discrete text generation with an autoregressive sequence of continuous "latent thinking tokens", enabling the retriever to "think silently" while preserving rich semantics and full differentiability.
- ♊ **Dual-View Training Paradigm**: A unified "Teacher-Student" architecture where the **Explicit-View** acts as a semantic mentor with Chain-of-Thought (CoT) rationales, while the **Latent-View** learns to perform implicit reasoning internally under its guidance.
- 🎯 **Multi-Grained Alignment**:
  - *Output-level Distillation*: Synchronizes the final ranking preferences between the explicit and latent views.
  - *Process-level Trajectory Alignment*: Utilizes temporal downsampling to ensure the latent tokens accurately capture the step-by-step semantic progression of explicit reasoning paths.
- ⚡ **Uncompromising Efficiency**: By restricting the thinking process to a compact horizon of latent tokens (typically K < 10), LaSER achieves complex reasoning capabilities while incurring only ~0.3% of the latency of "rewrite-then-retrieve" pipelines (~1.7× overhead over standard single-pass dense retrievers).
- 🌐 **Exceptional Versatility**: Seamlessly compatible with diverse LLM backbones across various scales (0.6B to 8B parameters), delivering consistent gains across both in-domain and out-of-domain benchmarks.

## 🧠 Framework

![Framework](<./assets/main_arch.png>)

The LaSER framework operates on a shared LLM backbone with two views:
- **Explicit-View (Training only)**: Encodes the query along with a high-quality CoT rationale. Intermediate hidden states at reasoning segment boundaries are extracted via temporal downsampling (M → K features).
- **Latent-View (Training & Inference)**: Generates K continuous latent thinking tokens autoregressively — each step projects the last hidden state through the LM head, computes a soft token as the probability-weighted embedding, and appends it to the sequence. The final representation is the mean-pool of all K hidden states.

A multi-grained self-distillation objective aligns these two views at both the output level and the intermediate trajectory level. At inference time, **only the Latent-View is used**, eliminating the need for slow text generation.

## 📊 Overall Performance

LaSER significantly outperforms state-of-the-art baselines across multiple reasoning-intensive benchmarks, including **BRIGHT** (in-domain), **BrowseComp-Plus** (out-of-domain), and **FollowIR** (out-of-domain).

![Performance on BRIGHT](<./assets/overall_performance.png>)

### Main Results on BRIGHT (nDCG@10)

| Model | Size | Bio. | Earth. | Econ. | Psy. | Rob. | Stack. | Sus. | Leet. | Pony | AoPS | TheoQ. | TheoT. | **Avg.** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-Embedding | 0.6B | 12.7 | 26.3 | 17.9 | 16.5 | 12.5 | 12.4 | 12.2 | 14.3 | 0.7 | 3.1 | 17.2 | 26.5 | 14.4 |
| Qwen3-Embedding | 8B | 14.7 | 17.9 | 15.5 | 19.9 | 9.1 | 12.9 | 16.5 | 17.4 | 0.8 | 2.5 | 16.8 | 24.5 | 14.0 |
| Fair Baseline | 0.6B | 28.8 | 31.6 | 25.2 | 28.1 | 15.1 | 22.3 | 24.2 | 8.8 | **3.5** | 2.1 | 14.1 | 15.4 | 18.3 |
| Fair Baseline | 8B | 49.7 | 51.2 | 26.9 | 37.4 | 23.4 | 28.0 | 34.1 | 3.7 | 3.2 | 2.8 | 16.8 | 31.8 | 25.7 |
| Rewrite-then-Retrieve † | 0.6B | 57.8 | 51.6 | 14.1 | 39.4 | 15.0 | 19.8 | 23.7 | 0.6 | 14.6 | 1.2 | 14.9 | 15.5 | 22.4 |
| Rewrite-then-Retrieve † | 8B | 53.1 | 54.3 | 32.1 | 34.8 | 20.5 | 31.1 | 32.2 | 3.2 | 15.2 | 4.1 | 17.4 | 38.8 | 28.1 |
| GIRCSE | 0.6B | 29.0 | 32.8 | 24.7 | 30.6 | 13.6 | 24.0 | 26.6 | 11.1 | 1.1 | 1.3 | 13.5 | 20.6 | 19.1 |
| GIRCSE | 8B | **59.0** | **56.5** | 27.2 | 40.3 | 19.0 | 28.5 | 31.4 | 3.2 | 3.6 | 1.7 | 14.0 | 27.2 | 26.0 |
| **LaSER (Ours)** | **0.6B** | 50.0 | 45.9 | 25.7 | 32.4 | 18.4 | 27.1 | 26.5 | 9.1 | 2.7 | 1.2 | 16.0 | 22.4 | **23.1** |
| **LaSER (Ours)** | **8B** | 58.4 | 48.1 | **28.0** | **40.9** | **17.0** | **29.9** | **28.3** | 1.7 | 5.9 | 1.5 | **14.6** | **19.2** | **29.3** |

> † indicates methods that use an external LLM to rewrite queries during inference.

### Performance Across Model Scales

![Performance of Various Sizes](<./assets/performance_3.png>)

## 🤗 Model Zoo

We release LaSER models in three sizes, all based on the Qwen3 backbone:

| Model | Size | Backbone | BRIGHT (nDCG@10) | HuggingFace |
|:---|:---:|:---|:---:|:---:|
| **LaSER-Qwen3-0.6B** | 0.6B | Qwen/Qwen3-0.6B | 23.1 | [🤗 Link](https://huggingface.co/Alibaba-NLP/LaSER-Qwen3-0.6B) |
| **LaSER-Qwen3-4B** | 4B | Qwen/Qwen3-4B | 28.0 | [🤗 Link](https://huggingface.co/Alibaba-NLP/LaSER-Qwen3-4B) |
| **LaSER-Qwen3-8B** | 8B | Qwen/Qwen3-8B | 29.3 | [🤗 Link](https://huggingface.co/Alibaba-NLP/LaSER-Qwen3-8B) |

## 🚀 Usage

### Installation

```bash
git clone https://github.com/ignorejjj/LaSER.git
cd LaSER
pip install -r requirements.txt
```

### Quick Start

Generate query/document embeddings using our model:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def laser_encode(model, tokenizer, texts, max_length=512, num_thinking_steps=3):
    """Encode texts using LaSER's latent thinking mechanism."""
    device = next(model.parameters()).device
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    batch_size = input_ids.size(0)
    thinking_slots = num_thinking_steps - 1
    eos_id = tokenizer.eos_token_id

    # Pad with EOS tokens for thinking slots
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

    # Autoregressive latent thinking
    for step_idx in range(thinking_slots):
        pos = base_seq_len + step_idx
        step_embeds = input_embeds[:, :pos, :] if past_key_values is None else input_embeds[:, pos-1:pos, :]
        step_mask = attention_mask[:, :pos]

        outputs = model(inputs_embeds=step_embeds, attention_mask=step_mask,
                       output_hidden_states=True, past_key_values=past_key_values,
                       use_cache=True, return_dict=True)

        hidden_steps.append(outputs.hidden_states[-1][:, -1, :])
        # Soft token: probability-weighted embedding
        token_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        new_embed = token_probs @ embedding_table
        past_key_values = outputs.past_key_values

        pre = input_embeds[:, :pos, :]
        post = input_embeds[:, pos+1:, :]
        input_embeds = torch.cat([pre, new_embed.unsqueeze(1), post], dim=1)

    # Final step
    final_embeds = input_embeds[:, -1:, :] if past_key_values else input_embeds
    outputs = model(inputs_embeds=final_embeds, attention_mask=attention_mask,
                   output_hidden_states=True, past_key_values=past_key_values,
                   use_cache=True, return_dict=True)
    hidden_steps.append(outputs.hidden_states[-1][:, -1, :])

    # Mean-pool all thinking step hidden states and normalize
    embeddings = torch.stack(hidden_steps, dim=1).mean(dim=1)
    return F.normalize(embeddings, p=2, dim=-1)


# Load model
model_name = "Alibaba-NLP/LaSER-Qwen3-0.6B"  # or LaSER-Qwen3-4B, LaSER-Qwen3-8B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
).cuda().eval()

# Encode
with torch.inference_mode():
    query_emb = laser_encode(model, tokenizer, ["why is the sky blue"], num_thinking_steps=3)
    doc_emb = laser_encode(model, tokenizer, ["Rayleigh scattering makes short wavelengths scatter more strongly"], num_thinking_steps=3)

# Compute similarity
similarity = (query_emb @ doc_emb.T).item()
print(f"Cosine similarity: {similarity:.4f}")
```

You can also use the provided script:

```bash
python src/scripts/quick_start.py \
  --model_path Alibaba-NLP/LaSER-Qwen3-0.6B \
  --query "why is the sky blue" \
  --doc "Rayleigh scattering makes short wavelengths scatter more strongly" \
  --num_thinking_steps 3
```

### Training

LaSER training is built on a vendored [Tevatron](https://github.com/TextTron/Tevatron) codebase.

#### 1) Prepare Training Data

`TrainDatasetWithRewrite` expects a HuggingFace dataset saved with `datasets.save_to_disk(...)`, featuring the following fields:

| Field | Description |
|:---|:---|
| `query` | The original query text |
| `reasoning_query` | CoT rationale generated by an external reasoner (e.g., GPT-4o-mini) |
| `prompt` | Task-specific instruction |
| `positive_passages` | List of relevant documents |
| `negative_passages` | List of hard negative documents |

Download our pre-processed datasets directly from HuggingFace:

```bash
mkdir data
huggingface-cli download jinjiajie/LaSER-Training --local-dir ./data/ --repo-type dataset
```

#### 2) Launch LaSER Training

```bash
bash ./scripts/train_laser_qwen3.sh
```

<details>
<summary>Key training hyperparameters</summary>

| Parameter | Value |
|:---|:---|
| Base model | Qwen/Qwen3-0.6B |
| LoRA rank / alpha | 64 / 32 |
| LoRA target modules | q, k, v, o, up, down, gate projections |
| Batch size per device | 8 |
| Gradient accumulation | 2 steps |
| Learning rate | 1e-4 |
| Warmup ratio | 0.1 |
| Epochs | 1 |
| Max seq length (query & passage) | 512 |
| Latent thinking steps (K) | 3 |
| Temperature (τ) | 0.02 |
| Loss weights (λ₁, λ₂, λ₃) | 1, 10, 0.1 |
| Training data | 81K examples from [ReasonEmb](https://huggingface.co/datasets/reasonir/ReasonEmb) |

</details>

#### 3) Merge LoRA Checkpoints

```bash
python src/scripts/merge_lora.py \
  --base-model-path Qwen/Qwen3-0.6B \
  --lora-path ./outputs/laser-qwen3-0.6b/checkpoint-XXXX
```

The merged model weights will be saved in the `lora-path` directory with a `-merged` suffix.

### Evaluation

Our evaluation is based on the [MTEB](https://github.com/embeddings-benchmark/mteb) benchmark. Run evaluation on reasoning-intensive benchmarks:

```bash
bash scripts/run_eval.sh
```

Summarize the results:

```bash
python eval/summary.py results/mteb/${model_name}/${model_name}/no_version_available ${benchmark_name}
```

> **Supported benchmarks:** `BRIGHT`, `FollowIR`, `BrowseCompPlus`

## 📂 Project Structure

```
LaSER/
├── requirements.txt
├── assets/                              # Figures and paper
├── src/
│   ├── eval/
│   │   ├── laser_model.py              # MTEB-compatible LaSER inference wrapper
│   │   ├── run_mteb.py                 # MTEB evaluation runner
│   │   └── summary.py                  # Result summarization
│   ├── scripts/
│   │   ├── quick_start.py             # Standalone inference demo
│   │   ├── train_laser_qwen3.sh       # Training launch script
│   │   ├── run_eval.sh                # Evaluation script
│   │   ├── merge_lora.py             # LoRA → full model merger
│   │   └── ds_stage0.json            # DeepSpeed config
│   └── tevatron/                       # Vendored & modified Tevatron framework
│       └── retriever/
│           ├── modeling/
│           │   └── laser.py           # ★ Core LaSER model implementation
│           ├── driver/
│           │   └── train_laser.py     # LaSER training entry point
│           ├── dataset.py             # Dataset classes (TrainDatasetWithRewrite)
│           ├── collator.py            # Data collation with rewrite support
│           ├── trainer.py             # Custom trainer
│           └── arguments.py           # Argument definitions
```

## 🙏 Acknowledgement

We sincerely thank the developers of [MTEB](https://github.com/embeddings-benchmark/mteb), [Tevatron](https://github.com/TextTron/Tevatron), and [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) for their foundational open-source contributions and support.

## 📝 Citation

If you find our code or paper useful, please cite our work:

```bibtex
@inproceedings{jin2026laser,
  title={LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval},
  author={Jin, Jiajie and Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Xie, Pengjun and Zhu, Yutao and Dou, Zhicheng},
  year={2026},
  url={https://arxiv.org/abs/2603.01425},
}
```

# <div align="center"> LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval </div>

<div align="center">
<a href="https://arxiv.org/abs/xxxx" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv"></a>
<a href="https://huggingface.co/datasets/jinjiajie/LaSER-Training" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg"></a>
<a href="https://github.com/ignorejjj/LASER"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>


---

## 📖 Introduction
**LaSER** is a novel self-distillation framework designed to bridge the gap between powerful but slow explicit reasoning (like Chain-of-Thought) and efficient but shallow dense retrieval. While Large Language Models (LLMs) possess strong reasoning capabilities, current retrievers often treat them as static encoders, leaving their reasoning potential dormant. 

![Introduction](<./assets/intro.png>)

Existing "rewrite-then-retrieve" pipelines suffer from prohibitive latency due to autoregressive text generation. In contrast, LaSER internalizes explicit reasoning into the **latent space**, allowing the retriever to "think" silently through continuous latent tokens. This enables the model to successfully combine the reasoning depth of explicit CoT pipelines with the inference efficiency of standard dense retrievers.
## ✨ Key Features

- 🧠 **Latent Thinking Mechanism**: **Say goodbye to slow text generation!** LaSER innovatively replaces discrete text generation with an autoregressive sequence of continuous "latent thinking tokens." This empowers the retriever to "think silently," preserving rich semantics and full differentiability without the latency bottleneck.
- ♊ **Dual-View Training Paradigm**: **A unified "Teacher-Student" architecture.** The **Explicit-View** acts as a semantic mentor equipped with privileged Chain-of-Thought (CoT) rationales, while the **Latent-View** learns to seamlessly perform implicit reasoning internally under its guidance.
- 🎯 **Multi-Grained Alignment**: **Matching both the destination and the journey!**
  - 🏆 *Output-level Distillation*: Perfectly synchronizes the final ranking preferences between the explicit and latent views.
  - 🛤️ *Process-level Trajectory Alignment*: Utilizes innovative temporal downsampling to ensure the latent tokens accurately capture the step-by-step semantic progression of explicit reasoning paths.
- ⚡ **Uncompromising Efficiency**: **The best of both worlds!** By restricting the thinking process to a compact horizon of latent tokens (typically $K < 10$), LaSER achieves complex reasoning capabilities while maintaining the lightning-fast inference latency of standard single-pass dense retrievers.
- 🌐 **Exceptional Versatility**: **Robust and highly scalable.** Seamlessly compatible with diverse LLM backbones across various scales (e.g., 0.6B to 8B parameters). Delivers rock-solid performance gains across both in-domain and rigorous out-of-domain benchmarks.

## 🧠 Framework

![Framework](<./assets/main_arch.png>)

The LaSER framework operates on a shared LLM backbone. During training, the Explicit-View encodes the original query along with a high-quality CoT rationale. Simultaneously, the Latent-View generates $K$ continuous latent tokens to simulate this reasoning process internally. A self-distillation objective aligns these two views at both the final representation level and the intermediate trajectory level. At inference time, only the Latent-View is used, eliminating the need for slow text generation.

## 📊 Overall Performance
LaSER significantly outperforms state-of-the-art baselines across multiple reasoning-intensive benchmarks, including **BRIGHT**, **BrowseComp-Plus**, and **FollowIR**.

- **Superior Accuracy**: It achieves performance comparable to, and in some cases better than, computationally intensive "rewrite-then-retrieve" pipelines.
- **Robust Efficiency**: As shown in our latency-performance analysis, LaSER maintains high retrieval quality while avoiding the exponential latency increase associated with autoregressive rewriters.
- **Generalizability**: The framework shows consistent gains across various model scales (e.g., 0.6B to 8B parameters).

![Performance on Bright](<./assets/overall_performance.png>)

![Performance of Various Sizes](<./assets/performance_3.png>)

## 🚀 Usage

### Installation

```bash
git clone [https://github.com/ignorejjj/LASER.git](https://github.com/ignorejjj/LASER.git)
cd LASER
pip install -r requirements.txt

```

### Training

LaSER training is built on a vendored Tevatron codebase.

#### 1) Prepare training data

`TrainDatasetWithRewrite` expects a HuggingFace dataset saved with `datasets.save_to_disk(...)`, featuring the following fields:

* `query`
* `reasoning_query`
* `prompt`
* `positive_passages`
* `negative_passages`

You can download our pre-processed datasets directly from HuggingFace and place them in the `data/` directory:

```bash
mkdir data
hf download jinjiajie/LaSER-Training --local-dir ./data/ --repo-type dataset 
```

#### 2) Launch LaSER training

To train the LaSER model, run the following script (please modify the paths and parameters inside the script according to your environment):

```bash
bash ./scripts/train_laser_qwen3.sh

```

#### 3) Merge LoRA checkpoints

```bash
python src/scripts/merge_lora_new.py \
  --base-model-path Qwen/Qwen3-0.6B \
  --lora-path ./outputs/laser-qwen3-0.6b/checkpoint-XXXX \

```

The merged model weights will be saved in the `lora-path` directory with a `-merged` suffix.

### Evaluation

Our evaluation is based on the MTEB benchmark. You can use the following command to run the evaluation (please ensure you update the paths in the script first):

```bash
bash scripts/run_eval.sh 

```

Once the evaluation is complete, you can summarize the results using the following command:

```bash
python eval/summary.py results/mteb/${model_name}/${model_name}/no_version_available ${benchmark_name} 

```

> **Note:** The `benchmark_name` can be reasoning-intensive datasets supported by MTEB, such as `BRIGHT` or `FollowIR`. `BrowseComp-Plus` is not yet officially supported by MTEB; we will release the corresponding evaluation scripts for it shortly.

## 🙏 Acknowledgement

We sincerely thank the developers of [MTEB](https://github.com/embeddings-benchmark/mteb), [Tevatron](https://github.com/TextTron/Tevatron), and [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) for their foundational open-source contributions and support.

## 📝 Citation

If you find our code or paper useful, please cite our work:

```bibtex
@misc{jin2026laser,
  title={LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval},
  author={Jin, Jiajie and Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Xie, Pengjun and Zhu, Yutao and Dou, Zhicheng},
  year={2026},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/}, 
}
```


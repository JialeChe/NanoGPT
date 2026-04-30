<h1 align="center">NanoGPT</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">

</p>

<p align="center">
  <b>从零构建 Decoder-Only GPT · 逐步集成现代 LLM 架构 </b>
</p>

## 已实现

| # | 技术 | 位置 | 简介 |
|---|------|------|------|
| 1 | **基础 GPT** | `model/decoder_transformer.py` | Token Embedding + Causal MHA + GELU FFN |
| 2 | **RoPE** | `model/utils.py` | Q/K 旋转变换|
| 3 | **RMSNorm** | `model/component.py::rmsnorm` | 仅 RMS 归一化，比 LayerNorm 更快 |
| 4 | **SwiGLU** | `model/component.py::ffn_swiglu` | `SiLU(gate) ⊙ up` 门控激活 |
| 5 | **MoE** | `model/component.py::moe` | Top-K 稀疏专家混合，扩大容量不增计算 |

---

## 📦 数据准备

### 预训练数据

下载 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，将 `TinyStoriesV2-GPT4-train.txt` 放入 `dataset/`。

### SFT 数据

JSON 格式，放入 `sft_data/`：

```json
[
  {
    "instruction": "翻译以下句子为英文",
    "input": "你好世界",
    "output": "hello world"
  }
]
```
---

## 📋 TODO

- [ ] DeepSpeed 分布式训练加速（或纯 PyTorch DDP / FSDP）
- [ ] KV-Cache 推理优化
- [ ] 其他 MoE 变体（DeepSeek 共享专家 / 细粒度专家等）
- [ ] Attention Residuals--Kimi

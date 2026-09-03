# transformer

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个**从零实现**的 Byte-level BPE Decoder-only Transformer，用于在 Tiny Shakespeare 语料上学习"下一个 token"预测，并逐 token 续写文本。

项目的目标是把《Attention Is All You Need》中的核心思想落实到可以运行的代码：**不使用 `torch.nn.Transformer`、`nn.MultiheadAttention` 等封装**，而是用 `nn.Linear`、`nn.LayerNorm` 等基础层，自己写出 tokenizer、多头注意力、前馈网络、残差连接和训练循环，每一处都可直接阅读与修改。

## 项目特点

- 手写 **byte-level BPE tokenizer**：从 UTF-8 字节出发，反复合并最高频相邻 pair，得到固定大小的词表；支持训练、编码、解码与 JSON 持久化；
- 手写 **Decoder-only Transformer**：多头因果自注意力（显式上三角 mask）、前馈网络、Post-LN 残差块、正弦位置编码；
- 手写**训练与生成流程**：随机连续片段 batch、交叉熵损失、Adam 优化、梯度裁剪、贪心逐 token 生成；
- **可复现**：随机种子固定、tokenizer merge 次序确定性、checkpoint 同时保存权重/配置/tokenizer merges，加载即可重建完全相同的模型；
- 单文件 `transformer.py` 覆盖全部功能，提供 `train-tokenizer` / `train` / `generate` 三个 CLI 子命令；
- 自动选择设备：优先 MPS（Apple Silicon），否则 CPU。

## 实现说明

代码完全自己实现，不使用 PyTorch 封装好的 Transformer 或 Attention 组件：

| 模块 | 实现 |
| --- | --- |
| Byte-level BPE tokenizer | `ByteBPETokenizer`（train / encode / decode / save / load） |
| 多头因果自注意力 | `MultiHeadSelfAttention`（线性投影 Q/K/V，缩放点积，上三角 mask） |
| 前馈网络与残差块 | `FeedForward`、`TransformerBlock`（残差 + Post-LN） |
| Decoder-only Transformer | `DecoderOnlyTransformer`（token 嵌入 + 正弦位置编码 + 多层堆叠） |
| 数据划分与 batch 构造 | `split_data`、`get_batch`（90% 训练 / 10% 验证） |
| 评估与训练循环 | `evaluate`、`train`（保存验证集最优 checkpoint） |
| 贪心生成 | `generate`（每次只把最新上下文送入模型） |
| 命令行入口 | `main`（`train-tokenizer` / `train` / `generate`） |

模型前向流程：

```text
token IDs ──token_embedding──▶ ┐
                              (+)──▶ 残差块 × num_layers ──▶ Linear ──▶ (B, T, V) logits
位置索引 ──正弦位置编码─────────┘   │
                                   ├── 因果多头自注意力（只看左侧上下文）
                                   └── 前馈网络（逐 token 加工）＋ 残差连接 / Post-LN
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+（`torch.load(..., weights_only=True)` 需要 2.0 及以上）
- Apple Silicon 可选：MPS 加速（无 MPS 时自动退回 CPU）

```bash
pip install torch
```

## 快速开始

仓库自带 Tiny Shakespeare 语料（`data/input.txt`）与训练好的 tokenizer、checkpoint，可直接生成：

```bash
python transformer.py generate \
  --checkpoint checkpoints/model.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 200
```

从零复现完整流程：

```bash
# 1. 训练 tokenizer（默认 512 词表，输出 tokenizer.json）
python transformer.py train-tokenizer --vocab-size 512

# 2. 训练模型（5000 步，每 250 步评估一次并保存验证集最优 checkpoint）
python transformer.py train --max-steps 5000 --eval-interval 250

# 3. 用训练好的 checkpoint 贪心续写
python transformer.py generate --checkpoint checkpoints/model.pt --prompt "ROMEO:"
```

## 命令行参数

### `train-tokenizer`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--vocab-size` | `512` | 目标词表大小（含 256 个 byte token，不能小于 256） |
| `--data` | `data/input.txt` | 训练语料路径 |
| `--output` | `tokenizer.json` | tokenizer 输出路径 |

### `train`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max-steps` | `5000` | 最大训练步数 |
| `--eval-interval` | `250` | 每隔多少步评估一次 |
| `--data` | `data/input.txt` | 训练语料路径 |
| `--tokenizer` | `tokenizer.json` | 已训练的 tokenizer 路径 |
| `--checkpoint` | `checkpoints/model.pt` | checkpoint 保存路径 |

### `generate`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | 必填 | 待加载的 checkpoint 路径 |
| `--prompt` | `ROMEO:` | 生成起始文本 |
| `--max-new-tokens` | `500` | 最多续写 token 数 |

所有默认路径均相对于脚本自身定位，因此在任意目录执行 `python transformer/transformer.py ...` 都能找到对应文件。

## 默认超参数（`Config`）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `context_length` | 64 | 最大上下文 / 序列长度 |
| `d_model` | 64 | 嵌入维度 |
| `num_heads` | 4 | 注意力头数（d_model 须可整除） |
| `num_layers` | 3 | Transformer 层数 |
| `d_ff` | 256 | 前馈网络隐藏维度 |
| `batch_size` | 32 | 训练 batch 大小 |
| `eval_batches` | 20 | 每次评估的随机批次数量 |
| `seed` | 1337 | 随机种子 |

训练时每 `eval_interval` 步打印一次训练/验证损失，验证损失更低的 checkpoint 才会被保存：

```text
step 250: train_loss=..., val_loss=...
step 500: train_loss=..., val_loss=...
```

## 生成示例

使用仓库自带 checkpoint 的贪心输出（`prompt="ROMEO:"`，120 token）：

```text
ROMEO: OF GAUNT:
I'll prove you, sir, sir, what we will deful
By the world of the world of death.

KING RICHARD II:
What was we will not so much of your kindness.

KING RICHARD III:
What was we will not so much of your kindness.
```

## 项目结构

```text
transformer/
├── README.md                       # 本文件
├── LICENSE                         # MIT License
├── .gitignore
├── transformer.py                  # 全部实现：tokenizer / 模型 / 训练 / 生成 / CLI
├── tokenizer.json                  # 训练好的 byte-level BPE（vocab_size=512）
├── data/
│   └── input.txt                   # Tiny Shakespeare 语料（默认训练数据）
└── checkpoints/
    └── model.pt                    # 验证集最优 checkpoint（含权重/配置/tokenizer merges）
```

## 参考

- Vaswani et al., [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762)
- Tiny Shakespeare 语料：Andrej Karpathy 的 [char-rnn 数据目录](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)

## 贡献

欢迎通过 **Issue** 报告问题或提出改进建议，也欢迎提交 **Pull Request**：

- 提交前请先阅读 `transformer.py` 与相关 Issue，说明改动动机；
- 本项目定位是**教学**仓库，代码以"可读性优先"——新功能请尽量沿用手写实现风格，保持单文件可整体阅读，并在文档字符串中说明输入/输出与设计意图；
- 如有 API 或行为变化，请同步更新本 README。

## 许可证

本项目代码采用 [MIT License](LICENSE) 开源。

- 语料（`data/input.txt`）为 Tiny Shakespeare 公开语料，来源见上文[参考](#参考)，莎士比亚作品本身属于公有领域。

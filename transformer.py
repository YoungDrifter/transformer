"""从 Tiny Shakespeare 学习下一个 BPE token 的 Decoder-only Transformer。"""

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import torch
import torch.nn.functional as F
from torch import nn

# ==================== 1. 配置：集中放置可调训练参数 ====================
# 训练、保存和加载都使用同一个 Config，因此 checkpoint 能重建完全相同的模型。
@dataclass
class Config:
    """保存模型与训练参数；输入为可选配置值，输出为配置对象，供全流程共享。"""
    context_length: int = 64  # Transformer 的最大上下文长度；每个训练批次的序列长度。
    d_model: int = 64  # Transformer 的嵌入维度；每个字符的向量表示长度。
    num_heads: int = 4  # Transformer 的注意力头数。
    num_layers: int = 3  # Transformer 的层数。
    d_ff: int = 256  # 前馈网络的隐藏层维度。
    batch_size: int = 32  # 训练批次大小。
    max_steps: int = 5000  # 最大训练步数。
    eval_interval: int = 250  # 评估间隔。
    eval_batches: int = 20  # 每次评估的批次数量。
    seed: int = 1337  # 随机种子.


def select_device() -> torch.device:
    """选择计算设备；输入为空，输出优先为 MPS、否则为 CPU。"""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==================== 2. 文本与 tokenizer：BPE token 和整数 ID 互相转换 ====================
# Transformer 只接收整数 ID；byte-level BPE 先保证任意 UTF-8 文本都可编码，再合并高频 byte 组合。
def load_text(path: Path) -> str:
    """读取 UTF-8 语料；输入为文本路径，输出为完整 Tiny Shakespeare 字符串。"""
    return Path(path).read_text(encoding="utf-8")

Pair = tuple[int, int]

def merge_pair(token_ids: list[int], pair: Pair, new_token_id: int) -> list[int]:
    """把序列中互不重叠的目标 pair 从左到右替换为一个新 token。"""
    merged: list[int] = []
    index = 0
    while index < len(token_ids):
        if index + 1 < len(token_ids) and (token_ids[index], token_ids[index + 1]) == pair:
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


class ByteBPETokenizer:
    """从 UTF-8 bytes 出发，通过有序 BPE merges 完成编码和解码。"""

    def __init__(self, merges: Iterable[Pair] = ()) -> None:
        self.merges = tuple((int(left), int(right)) for left, right in merges)
        self._token_bytes: dict[int, bytes] = {
            token_id: bytes([token_id]) for token_id in range(256)
        }
        for offset, (left, right) in enumerate(self.merges):
            new_token_id = 256 + offset
            if left not in self._token_bytes or right not in self._token_bytes:
                raise ValueError("merge 引用了尚未创建的 token ID")
            self._token_bytes[new_token_id] = self._token_bytes[left] + self._token_bytes[right]

    @property
    def vocab_size(self) -> int:
        """返回 256 个 byte token 加全部 merge token 的总数。"""
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "ByteBPETokenizer":
        """统计相邻 token pair，反复合并最高频 pair，得到指定大小的词表。"""
        if vocab_size < 256:
            raise ValueError("vocab_size 不能小于 256，因为所有 byte 都必须在词表中")

        token_ids = list(text.encode("utf-8"))
        merges: list[Pair] = []
        while 256 + len(merges) < vocab_size:
            pair_counts = Counter(zip(token_ids, token_ids[1:]))
            # 次数相同时选择 ID 更小的 pair，使相同语料每次得到完全相同的 merges。
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            token_ids = merge_pair(token_ids, best_pair, 256 + len(merges))
            merges.append(best_pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        """输入任意 Unicode 文本，输出应用全部 merges 后的 token ID。"""
        token_ids = list(text.encode("utf-8"))
        for offset, pair in enumerate(self.merges):
            token_ids = merge_pair(token_ids, pair, 256 + offset)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """把 token ID 展开为 bytes，再严格解码回 UTF-8 文本。"""
        pieces: list[bytes] = []
        for token_id in token_ids:
            if token_id not in self._token_bytes:
                raise ValueError(f"未知 token ID: {token_id}")
            pieces.append(self._token_bytes[token_id])
        return b"".join(pieces).decode("utf-8")

    def save(self, path: Path) -> Path:
        """把 merge 顺序写入 JSON；顺序决定每个新 token 的 ID。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ByteBPETokenizer":
        """从 JSON 恢复 tokenizer，并校验记录的词表大小。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte_bpe":
            raise ValueError("tokenizer 文件类型不是 byte_bpe")
        tokenizer = cls(tuple(tuple(pair) for pair in payload["merges"]))
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("tokenizer 文件中的 vocab_size 与 merges 不一致")
        return tokenizer

def split_data(token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """划分语料；输入全部 ID，输出前 90% 训练张量和后 10% 验证张量。"""
    split_at = max(1, min(int(len(token_ids) * 0.9), len(token_ids) - 1))
    return (
        torch.tensor(token_ids[:split_at], dtype=torch.long),
        torch.tensor(token_ids[split_at:], dtype=torch.long),
    )

def get_batch(data: torch.Tensor, config: Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """抽取随机连续片段；输入一维 ID，输出 (B,T) 的输入 x 与右移标签 y。"""
    # 从一维语料随机取 B 个起点；x 是 (B,T)，y 是整体右移一格的 (B,T)。
    starts = torch.randint(len(data) - config.context_length, (config.batch_size,))
    offsets = torch.arange(config.context_length)
    x = torch.stack([data[start + offsets] for start in starts]).to(device)
    y = torch.stack([data[start + offsets + 1] for start in starts]).to(device)
    return x, y

# ==================== 3. 模型：自己写出 Decoder-only Transformer 的计算流程 ====================
# 使用 Linear、LayerNorm 等基础层，不使用 PyTorch 封装好的 Transformer 或 Attention。

class MultiHeadSelfAttention(nn.Module):
    """让每个 token 读取自己和之前的 token；输入输出都是 (B,T,C)。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads

        # Q、K、V 都由当前 hidden 经过不同的线性变换得到。
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = hidden.shape

        # (B,T,C) -> (B,H,T,D)，其中 C = H * D。
        # 每个 head 都在更小的 D 维空间里分别计算注意力。
        query = self.query(hidden).view(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).transpose(1, 2)
        key = self.key(hidden).view(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).transpose(1, 2)
        value = self.value(hidden).view(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).transpose(1, 2)

        # 每个 query 与所有 key 做点积，得到 token 之间的影响分数 (B,H,T,T)。
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.d_head)

        # 上三角代表未来位置，把它们设为 -inf 后，softmax 会给这些位置分配 0 权重。
        future_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(future_mask, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B,H,T,T)

        # 用注意力权重加权 V，得到每个 token 汇总历史信息后的向量。
        context = attention_weights @ value  # (B,H,T,D)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )  # 把多个 head 重新拼回 (B,T,C)。
        return self.output(context)


class FeedForward(nn.Module):
    """逐个处理 token 向量；输入输出都是 (B,T,C)，token 之间不在这里交互。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.linear1(hidden))  # (B,T,C) -> (B,T,d_ff)
        return self.linear2(hidden)  # (B,T,d_ff) -> (B,T,C)


class TransformerBlock(nn.Module):
    """组合注意力与前馈网络，并用残差连接保留进入本层的信息。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # attention_output 给每个 token 加入左侧上下文，再与原 hidden 做残差相加。
        attention_output = self.attention(hidden)
        hidden = self.norm1(hidden + attention_output)

        # FFN 独立加工每个 token；第二次残差连接后得到本层最终输出。
        feed_forward_output = self.feed_forward(hidden)
        hidden = self.norm2(hidden + feed_forward_output)
        return hidden


class DecoderOnlyTransformer(nn.Module):
    """建立字符语言模型；输入 (B,T) ID，输出每个位置的 (B,T,V) logits。"""

    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        # 固定的 sin/cos 位置编码告诉模型每个 token 位于序列的什么位置。
        positions = torch.arange(config.context_length).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, config.d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / config.d_model)
        )
        position_encoding = torch.zeros(config.context_length, config.d_model)
        position_encoding[:, 0::2] = torch.sin(positions * frequencies)
        cosine_columns = position_encoding[:, 1::2].shape[1]
        position_encoding[:, 1::2] = torch.cos(
            positions * frequencies[:cosine_columns]
        )
        self.register_buffer("position_encoding", position_encoding)

        # ModuleList 让 PyTorch 能找到每一层参数；forward 中仍由我们自己逐层调用。
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.output = nn.Linear(config.d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """执行完整前向流程；(B,T) ID 最终变为 (B,T,V) 字符分数。"""
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("输入长度不能超过 context_length")
        token_vectors = self.token_embedding(token_ids)  # (B,T,C)：字符向量。
        position_vectors = self.position_encoding[:sequence_length]  # (T,C)
        hidden = token_vectors + position_vectors  # (B,T,C)：加入顺序信息。

        for block in self.blocks:
            hidden = block(hidden)  # 每层都执行 attention -> FFN。

        return self.output(hidden)  # (B,T,V)：下一个字符的未归一化分数。
    
# ==================== 4. 评估和训练：从随机批次得到损失并更新参数 ====================
# 训练让预测分布接近右移标签；评估只读取模型质量，不累积梯度也不更新权重。
def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算交叉熵；输入 (B,T,V) logits 与 (B,T) 标签，输出一个标量损失。"""
    # 展平 (B,T,V) 为 (B*T,V)，并展平右移 labels 为 (B*T)，逐字符计算交叉熵。
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def evaluate(
    model: DecoderOnlyTransformer, data: torch.Tensor, config: Config, device: torch.device
) -> float:
    """评估模型；输入模型、数据和配置，输出若干随机验证批次的平均损失。"""
    was_training = model.training
    model.eval()  # 切换到评估模式，表示这里只计算结果、不训练模型。
    try:
        with torch.no_grad():
            losses = []
            for _ in range(config.eval_batches):
                x, y = get_batch(data, config, device)
                losses.append(compute_loss(model(x), y).item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)

def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    config: Config,
    tokenizer: ByteBPETokenizer,
    step: int,
    best_val_loss: float,
) -> Path:
    """保存 checkpoint；输入训练状态，输出写入完成的 checkpoint 路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 保留权重、配置和字符顺序，加载时才能得到一致的 ID 语义。
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "config": asdict(config),
            "tokenizer_merges": [list(pair) for pair in tokenizer.merges],
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[DecoderOnlyTransformer, Config, ByteBPETokenizer, int, float]:
    """加载 checkpoint；输入路径和设备，输出模型、配置、tokenizer 与训练元数据。"""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    tokenizer = ByteBPETokenizer(checkpoint["tokenizer_merges"])
    model = DecoderOnlyTransformer(tokenizer.vocab_size, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, tokenizer, int(checkpoint["step"]), float(checkpoint["best_val_loss"])


def train(config: Config, data_path: Path, tokenizer_path: Path, checkpoint_path: Path) -> None:
    """训练模型；显式加载已训好的 BPE tokenizer，保存验证集最佳 checkpoint。"""
    device = select_device()
    torch.manual_seed(config.seed)
    text = load_text(data_path)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    train_data, validation_data = split_data(tokenizer.encode(text))
    model = DecoderOnlyTransformer(tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    best_val_loss = float("inf")

    for step in range(1, config.max_steps + 1):
        x, y = get_batch(train_data, config, device)
        logits = model(x)  # (B,T,V)，预测 x 每个位置右侧的字符。
        loss = compute_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()  # backward 反向传播，把损失对所有参数的梯度写入 .grad。
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 可选裁剪，抑制过大的梯度。
        optimizer.step()  # Adam 使用梯度更新嵌入、Transformer 和输出层参数。

        if step == 1 or step % config.eval_interval == 0 or step == config.max_steps:
            train_loss = evaluate(model, train_data, config, device)
            validation_loss = evaluate(model, validation_data, config, device)
            print(f"step {step}: train_loss={train_loss:.4f}, val_loss={validation_loss:.4f}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                save_checkpoint(checkpoint_path, model, config, tokenizer, step, best_val_loss)


# ==================== 5. 生成：加载后的模型逐字符贪心续写 ====================
# 每轮只将最新上下文送入模型，取最后位置分数最大的 ID，再把一个 token 追加回输入。
def generate(
    model: DecoderOnlyTransformer,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """贪心生成文本；输入模型、提示词和长度，输出包含提示词的续写字符串。"""
    if not prompt:
        raise ValueError("prompt 不能为空")
    token_ids = tokenizer.encode(prompt)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = token_ids[-model.config.context_length :]
                input_ids = torch.tensor([context], dtype=torch.long, device=device)  # (1,T)
                logits = model(input_ids)  # (1,T,V)，只读取最后一个时间步。
                next_token_id = int(logits[0, -1].argmax().item())
                token_ids.append(next_token_id)  # 一次只追加一个 token，再进入下一轮。
        return tokenizer.decode(token_ids)
    finally:
        model.train(was_training)


# ==================== 6. 命令行：把训练与生成连接到可直接运行的入口 ====================
# 默认路径从当前脚本定位，故无论在何处执行 python transformer/transformer.py 都能找到数据。
def build_parser() -> argparse.ArgumentParser:
    """构建 CLI；输入为空，输出含 train 和 generate 子命令的参数解析器。"""
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="byte-level BPE Decoder-only Transformer")
    commands = parser.add_subparsers(dest="command")

    tokenizer_parser = commands.add_parser("train-tokenizer", help="从语料训练并保存 byte-level BPE")
    tokenizer_parser.add_argument("--vocab-size", type=int, default=512)
    tokenizer_parser.add_argument("--data", type=Path, default=script_directory / "data" / "input.txt")
    tokenizer_parser.add_argument("--output", type=Path, default=script_directory / "tokenizer.json")

    train_parser = commands.add_parser("train", help="训练并保存验证集最优 checkpoint")
    train_parser.add_argument("--max-steps", type=int, default=5000)
    train_parser.add_argument("--eval-interval", type=int, default=250)
    train_parser.add_argument("--data", type=Path, default=script_directory / "data" / "input.txt")
    train_parser.add_argument("--tokenizer", type=Path, default=script_directory / "tokenizer.json")
    train_parser.add_argument("--checkpoint", type=Path, default=script_directory / "checkpoints" / "model.pt")

    generate_parser = commands.add_parser("generate", help="从 checkpoint 贪心生成文本")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", default="ROMEO:")
    generate_parser.add_argument("--max-new-tokens", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    """运行 CLI；输入可选参数列表，输出训练日志或生成文本。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train-tokenizer":
        tokenizer = ByteBPETokenizer.train(load_text(args.data), vocab_size=args.vocab_size)
        tokenizer.save(args.output)
        print(f"tokenizer 已保存到 {args.output}，vocab_size={tokenizer.vocab_size}")
    elif args.command == "train":
        config = Config(max_steps=args.max_steps, eval_interval=args.eval_interval)
        train(config, args.data, args.tokenizer, args.checkpoint)
    elif args.command == "generate":
        device = select_device()
        model, _, tokenizer, _, _ = load_checkpoint(args.checkpoint, device)
        print(generate(model, tokenizer, args.prompt, args.max_new_tokens, device))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

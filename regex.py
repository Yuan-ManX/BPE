import regex as re
from base import Tokenizer, get_stats, merge


# GPT-2 的分词模式，使用正则表达式定义
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-4 的分词模式，使用正则表达式定义
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    基于正则表达式的分词器实现。

    参数:
        pattern (str, 可选): 用于覆盖默认分词模式的字符串（默认使用 GPT-4 分词模式）。
        special_tokens (dict): 特殊标记的字典，键为标记字符串，值为对应的整数索引。
            例如: {'<|endoftext|>': 100257}
    """

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        """
        初始化 RegexTokenizer。

        1. 设置分词模式：
            - 如果未提供自定义模式，则使用 GPT-4 的默认分词模式。
            - 编译正则表达式模式以提高匹配效率。
        2. 初始化特殊标记字典及其反向字典。
        """
        super().__init__()
        # 设置分词模式，如果未提供自定义模式，则使用 GPT-4 的默认分词模式
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # 编译正则表达式模式以提高匹配效率
        self.compiled_pattern = re.compile(self.pattern)
        # 初始化特殊标记字典
        self.special_tokens = {}
        # 初始化反向特殊标记字典，用于快速查找
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        从输入文本中训练并构建词汇表。

        参数:
            text (str): 输入的训练文本。
            vocab_size (int): 期望的词汇表大小，必须大于或等于256。
            verbose (bool): 是否打印训练过程中的详细信息。

        异常:
            AssertionError: 如果词汇表大小小于256。
        """
        assert vocab_size >= 256
        # 需要进行的合并操作次数
        num_merges = vocab_size - 256

        # 使用正则表达式模式将文本拆分为文本块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 输入文本预处理
        # 将每个文本块编码为 UTF-8 字节，并转换为整数列表
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 初始化合并规则和词汇表
        # 合并规则字典，键为 (int, int) 类型的元组，值为合并后的整数索引
        merges = {} # (int, int) -> int
        # 初始化词汇表，前256个为单个字节
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # 迭代地进行合并操作
        for i in range(num_merges):
            # 统计每个连续字节对的出现次数
            stats = {}
            for chunk_ids in ids:
                # 传递 stats 字典以在原地更新统计信息，累加计数
                get_stats(chunk_ids, stats)
            # 找到出现次数最多的字节对
            pair = max(stats, key=stats.get)
            # 分配一个新的索引给这个合并后的字节对
            idx = 256 + i
            # 用新的索引替换文本中的所有该字节对
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 保存合并规则
            merges[pair] = idx
            # 更新词汇表
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 如果启用了详细模式，打印当前合并的信息
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存类变量，供后续的编码和解码使用
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        """
        注册特殊标记。

        参数:
            special_tokens (dict): 特殊标记的字典，键为标记字符串，值为对应的整数索引。
                例如: {"<|endoftext|>": 100257}
        """
        # 设置特殊标记字典
        self.special_tokens = special_tokens
        # 设置反向特殊标记字典，用于快速查找
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        将整数索引列表解码为原始字符串。

        参数:
            ids (List[int]): 需要解码的整数索引列表。

        返回:
            str: 解码后的字符串。
        """
        # 初始化字节列表
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                # 如果索引存在于词汇表中，则添加对应的字节
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # 如果索引存在于反向特殊标记字典中，则添加对应的特殊标记的字节
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                # 如果索引无效，则抛出异常
                raise ValueError(f"invalid token id: {idx}")
        # 将字节列表拼接成字节串
        text_bytes = b"".join(part_bytes)
        # 将字节串解码为字符串，遇到无法解码的字节则用替代字符替换
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """
        对输入的字节序列进行编码。

        参数:
            text_bytes (bytes): 输入的字节序列。

        返回:
            List[int]: 编码后的整数索引列表。
        """
        # 将所有字节转换为0..255范围内的整数
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到具有最低合并索引的字节对
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 如果没有更多的合并规则，则停止
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # 否则，合并最佳字节对（具有最低合并索引）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        """
        编码时忽略任何特殊标记。

        参数:
            text (str): 需要编码的字符串。

        返回:
            List[int]: 编码后的整数索引列表。
        """
        # 使用正则表达式模式将文本拆分为文本块
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # 分别编码每个文本块，然后将结果合并
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        与 encode_ordinary 不同，此函数处理特殊标记。

        参数:
            text (str): 需要编码的字符串。
            allowed_special (str|set): 可以是 "all"|"none"|"none_raise" 或自定义的特殊标记集合。

        返回:
            List[int]: 编码后的整数索引列表。

        异常:
            ValueError: 如果 allowed_special 参数的值无法识别。
            AssertionError: 如果 allowed_special="none_raise" 且文本中包含特殊标记。
        """
        # 解码用户对特殊标记处理的期望
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            # 确保文本中不包含任何特殊标记
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            # 仅允许指定的特殊标记
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # 如果没有特殊标记，则使用普通编码
            return self.encode_ordinary(text)
        # 否则，我们必须小心处理文本中潜在的特殊标记
        # 我们通过基于任何精确匹配的特殊标记的出现来拆分文本来处理特殊标记
        # 我们可以使用 re.split 来实现这一点。注意，将模式用 () 包围，使其成为捕获组，因此特殊标记将被包括在内
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # 现在所有特殊字符都与文本的其余部分分开
        # 分别编码每个文本块，然后将结果合并
        ids = []
        for part in special_chunks:
            if part in special:
                # 这是一个特殊标记，作为特殊情况单独编码
                ids.append(special[part])
            else:
                # 这是一个普通序列，正常编码
                ids.extend(self.encode_ordinary(part))
        return ids

import tiktoken
from regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    """
    辅助函数，用于在 get_gpt4_merges() 中重建合并森林。

    参数:
        mergeable_ranks (dict): 可合并的字节对及其排序字典，键为字节对（bytes），值为排序（int）。
        token (bytes): 需要进行 BPE 操作的字节序列。
        max_rank (int, 可选): 允许的最大排序值。如果设置为 None，则不限制。

    返回:
        list[bytes]: 合并后的字节列表。
    """
    # 将输入的字节序列拆分为单个字节的列表
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None  # 当前找到的最小索引
        min_rank = None  # 当前找到的最小排序值

        # 遍历所有可能的字节对
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # 获取当前字节对的排序值
            rank = mergeable_ranks.get(pair[0] + pair[1])

            # 如果当前字节对可以合并，并且其排序值小于当前的最小排序值
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i  # 更新最小索引
                min_rank = rank  # 更新最小排序值

        # 如果没有找到可以合并的字节对，或者当前找到的最小排序值不满足要求，则停止循环
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        # 执行合并操作：合并找到的字节对，并将结果放回列表中
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    # 返回合并后的字节列表
    return parts


def recover_merges(mergeable_ranks):
    """
    恢复合并规则，将字节序列的合并状态恢复为原始的字节对。

    参数:
        mergeable_ranks (dict): 可合并的字节对及其排序字典，键为字节对（bytes），值为排序（int）。

    返回:
        dict: 恢复后的合并规则字典，键为排序对（tuple），值为排序（int）。
    """
    # 初始化合并规则字典
    merges = {}

    # 遍历所有可合并的字节对
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # 跳过单个字节

        # 使用 BPE 算法对当前字节序列进行合并
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2

        # 恢复合并字节对的排序值
        ix0 = mergeable_ranks[pair[0]]  # 第一个字节的排序值
        ix1 = mergeable_ranks[pair[1]]  # 第二个字节的排序值
        # 将合并规则添加到字典中
        merges[(ix0, ix1)] = rank

    return merges


# GPT-4 的分词模式，使用正则表达式定义
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-4 的特殊标记及其对应的整数索引
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""
    """
    轻量级的 GPT-4 分词器封装，基于 RegexTokenizer 实现。
    """

    def __init__(self):
        """
        初始化 GPT4Tokenizer。
        
        1. 调用基类 RegexTokenizer 的初始化方法，并传入 GPT-4 的分词模式。
        2. 获取官方 GPT-4 分词器的合并规则。
        3. 恢复合并规则为原始的字节对。
        4. 从合并规则中重建词汇表。
        5. 处理字节重排（byte shuffle）。
        6. 注册特殊标记。
        """
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # 获取官方的 GPT-4 分词器及其合并规则
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks

        # 恢复合并规则为原始的字节对
        self.merges = recover_merges(mergeable_ranks)
        # 从合并规则中重建词汇表
        # 初始化词汇表，前256个为单个字节
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        # 处理字节重排（byte shuffle）
        # GPT-4 分词器对单个字节的标记进行了重排，这里需要处理这种重排
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # 注册特殊标记
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        """
        对输入的字节序列进行编码前的预处理：字节重排。
        
        参数:
            text_bytes (bytes): 输入的字节序列。
        
        返回:
            List[int]: 编码后的整数索引列表。
        """
        # 在处理字节之前，先进行字节重排
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        # 调用基类的编码方法进行编码
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        """
        解码整数索引列表为原始字符串。
        
        参数:
            ids (List[int]): 需要解码的整数索引列表。
        
        返回:
            str: 解码后的字符串。
        """
        # 先解码为字节序列
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # 进行逆字节重排
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        # 将字节序列解码为字符串，遇到无法解码的字节则用替代字符替换
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # GPT-4 分词器是一个预训练的分词器，不打算进行训练
    def train(self, text, vocab_size, verbose=False):
        """
        由于 GPT-4 分词器是预训练模型，因此不支持训练。
        
        参数:
            text (str): 输入的训练文本。
            vocab_size (int): 期望的词汇表大小。
            verbose (bool): 是否打印详细信息。
        
        异常:
            NotImplementedError: 不支持训练。
        """
        raise NotImplementedError

    # 保存和加载方法需要一些思考
    # 我们需要修改基类的保存和加载方法以支持字节重排
    # 或者我们可以将字节重排移到基类，但这会让我们的美丽的 Tokenizer 类变得丑陋
    # 只是为了支持 GPT-4 分词器和其奇怪的字节重排历史遗留问题
    def save(self, file_prefix):
        """
        由于 GPT-4 分词器不支持保存，因此抛出 NotImplementedError。
        
        参数:
            file_prefix (str): 文件名前缀。
        
        异常:
            NotImplementedError: 不支持保存。
        """
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        """
        由于 GPT-4 分词器不支持加载，因此抛出 NotImplementedError。
        
        参数:
            model_file (str): 模型文件路径。
        
        异常:
            NotImplementedError: 不支持加载。
        """
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        """
        为了可视化目的，将 GPT-4 的词汇表保存为与基类相同格式的文件。
        
        参数:
            vocab_file (str): 词汇表文件路径。
        
        文件内容:
            - 使用基类的 render_token 方法渲染每个标记。
            - 合并重排后的字节并写入文件。
        """
        from base import render_token
        # 构建词汇表，同时考虑字节重排
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            
        # 现在合并重排后的字节并写入文件
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

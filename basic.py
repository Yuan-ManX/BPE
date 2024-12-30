from base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """
    基本的分词器实现，使用字节对编码（BPE）算法。
    """

    def __init__(self):
        """
        初始化 BasicTokenizer。
        
        调用基类 Tokenizer 的初始化方法。
        """
        super().__init__()

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

        # 输入文本预处理
        # 将文本编码为原始字节
        text_bytes = text.encode("utf-8") # raw bytes
        # 将字节列表转换为整数列表，范围在0..255之间
        ids = list(text_bytes) # list of integers in range 0..255

        # 初始化合并规则和词汇表
        # 合并规则字典，键为 (int, int) 类型的元组，值为合并后的整数索引
        merges = {} # (int, int) -> int
        # 初始化词汇表，前256个为单个字节
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        # 迭代地进行合并操作
        for i in range(num_merges):
            # 统计每个连续字节对的出现次数
            stats = get_stats(ids)
            # 找到出现次数最多的字节对
            pair = max(stats, key=stats.get)
            # 分配一个新的索引给这个合并后的字节对
            idx = 256 + i
            # 用新的索引替换文本中的所有该字节对
            ids = merge(ids, pair, idx)
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

    def decode(self, ids):
        """
        将整数索引列表解码为原始字符串。
        
        参数:
            ids (List[int]): 需要解码的整数索引列表。
        
        返回:
            str: 解码后的字符串。
        """
        # 将词汇表中的字节拼接成原始字节串
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # 将字节串解码为字符串，遇到无法解码的字节则用替代字符替换
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        将输入字符串编码为整数索引列表。
        
        参数:
            text (str): 需要编码的字符串。
        
        返回:
            List[int]: 编码后的整数索引列表。
        """
        # 将文本编码为原始字节
        text_bytes = text.encode("utf-8") # raw bytes
        # 将字节列表转换为整数列表，范围在0..255之间
        ids = list(text_bytes) # list of integers in range 0..255

        # 当列表长度大于或等于2时，继续进行合并操作
        while len(ids) >= 2:
            # 找到可以合并的字节对中，合并索引最小的那个
            stats = get_stats(ids)
            # 使用 lambda 函数找到具有最小合并索引的字节对
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 如果没有更多的合并规则，则停止
            if pair not in self.merges:
                break # 无法再进行任何合并
            # 否则，合并最佳字节对（具有最小合并索引）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

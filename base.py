import unicodedata


def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回一个字典，该字典记录了连续对的出现次数。
    
    示例:
        输入: [1, 2, 3, 1, 2]
        输出: {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    
    可选地，允许更新一个已存在的计数字典。
    
    参数:
        ids (List[int]): 要统计的整数列表。
        counts (Dict[Tuple[int, int], int], 可选): 现有的计数字典，用于更新。
            默认为 None，表示创建一个新的字典。
    
    返回:
        Dict[Tuple[int, int], int]: 记录连续对出现次数的字典。
    """
    # 如果没有提供现有的 counts 字典，则初始化一个新的空字典
    counts = {} if counts is None else counts

    # 使用 zip 函数将列表中的元素与其后一个元素配对，生成连续的 (当前元素, 下一个元素) 对
    for pair in zip(ids, ids[1:]):
        # 更新 counts 字典中对应 pair 的计数
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    在整数列表 (ids) 中，将所有连续出现的 pair 替换为新的整数 token idx。
    
    示例:
        输入:
            ids = [1, 2, 3, 1, 2]
            pair = (1, 2)
            idx = 4
        输出:
            [4, 3, 4]
    
    参数:
        ids (List[int]): 要处理的整数列表。
        pair (Tuple[int, int]): 要被替换的连续对。
        idx (int): 替换后的新整数 token。
    
    返回:
        List[int]: 替换后的新整数列表。
    """
    # 初始化一个新的列表，用于存储替换后的结果
    newids = []
    # 初始化索引 i 为 0
    i = 0
    # 遍历整个 ids 列表
    while i < len(ids):
        # 检查当前索引 i 是否小于列表长度减一，以避免越界
        # 并且当前元素和下一个元素是否匹配要替换的 pair
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            # 如果匹配，则将新的 idx 添加到 newids 中
            newids.append(idx)
            # 跳过当前元素和下一个元素，因为它们已经被替换
            i += 2
        else:
            # 如果不匹配，则将当前元素添加到 newids 中
            newids.append(ids[i])
            # 移动到下一个元素
            i += 1
    # 返回替换后的新列表
    return newids


def replace_control_characters(s: str) -> str:
    """
    替换字符串中的控制字符，以避免输出时被扭曲（例如 \n 或更糟糕的情况）。
    
    参数:
        s (str): 输入的字符串。
    
    返回:
        str: 替换后的字符串，控制字符被转义为 Unicode 转义序列。
    """
    # 初始化一个空列表，用于存储处理后的字符
    chars = []
    for ch in s:
        # 获取字符的 Unicode 类别
        # 检查字符是否属于控制字符类别（Cc, Cf, Cs, Co, Cn）
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # 如果不是控制字符，则直接添加
        else:
            # 如果是控制字符，则将其转换为 Unicode 转义序列，例如 \u000a
            chars.append(f"\\u{ord(ch):04x}")
    # 将处理后的字符列表连接成一个字符串并返回
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    美化打印一个令牌，替换其中的控制字符以避免输出扭曲。
    
    参数:
        t (bytes): 输入的字节令牌。
    
    返回:
        str: 美化后的字符串，控制字符被转义为 Unicode 转义序列。
    """
    # 将字节解码为字符串，使用 'utf-8' 编码，如果遇到无法解码的字节则用替代字符替换
    s = t.decode('utf-8', errors='replace')
    # 替换字符串中的控制字符
    s = replace_control_characters(s)
    # 返回处理后的字符串
    return s


class Tokenizer:
    """Base class for Tokenizers"""
    """
    分词器的基础类。
    """

    def __init__(self):
        """
        初始化分词器。
        
        默认设置:
            - 词汇表大小为256（所有字节）
            - 无合并规则
            - 无特殊标记
        """
        # 合并规则字典，键为 (int, int) 类型的元组，值为合并后的整数索引
        self.merges = {} # (int, int) -> int
        # 模式字符串，用于定义分词规则（当前为空）
        self.pattern = "" # str

        # 特殊标记字典，键为特殊标记字符串，值为对应的整数索引
        # 例如: {'<|endoftext|>': 100257}
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        # 构建词汇表，键为整数索引，值为对应的字节串
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        """
        从输入文本中训练并构建一个大小为 vocab_size 的词汇表。
        
        参数:
            text (str): 输入的训练文本。
            vocab_size (int): 期望的词汇表大小。
            verbose (bool): 是否打印详细信息。
        
        异常:
            NotImplementedError: 子类需要实现此方法。
        """
        # 子类需要实现具体的训练逻辑
        raise NotImplementedError

    def encode(self, text):
        """
        将输入字符串编码为整数索引列表。
        
        参数:
            text (str): 需要编码的字符串。
        
        返回:
            List[int]: 编码后的整数索引列表。
        
        异常:
            NotImplementedError: 子类需要实现此方法。
        """
        # 子类需要实现具体的编码逻辑
        raise NotImplementedError

    def decode(self, ids):
        """
        将整数索引列表解码为原始字符串。
        
        参数:
            ids (List[int]): 需要解码的整数索引列表。
        
        返回:
            str: 解码后的字符串。
        
        异常:
            NotImplementedError: 子类需要实现此方法。
        """
        # 子类需要实现具体的解码逻辑
        raise NotImplementedError

    def _build_vocab(self):
        """
        构建词汇表。词汇表由合并规则和特殊标记决定。
        
        返回:
            Dict[int, bytes]: 构建好的词汇表，键为整数索引，值为对应的字节串。
        """
        # 初始化词汇表，256 个字节对应的词汇表项
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 根据合并规则更新词汇表
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 添加特殊标记到词汇表
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        保存分词器的模型和词汇表到文件中。
        
        文件命名:
            - file_prefix.model: 关键模型文件，用于 load() 方法加载
            - file_prefix.vocab: 仅供人类查看的词汇表文件
        
        文件内容:
            - model 文件包含版本信息、模式、特殊标记和合并规则
            - vocab 文件包含词汇表中每个索引对应的标记（可能包含部分 UTF-8 序列）
        
        参数:
            file_prefix (str): 文件名前缀。
        """
        # 写入模型文件
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本信息
            f.write("minbpe v1\n")
            # 写入模式字符串
            f.write(f"{self.pattern}\n")
            # 写入特殊标记的数量
            f.write(f"{len(self.special_tokens)}\n")
            # 写入每个特殊标记及其对应的索引
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 写入合并规则
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # 写入词汇表文件（仅供人类查看）
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # 注意：许多标记可能是部分的 UTF-8 序列，无法解码为有效的字符串。
                # 这里使用 errors='replace' 将其替换为替换字符。
                # 这也意味着我们不能使用 .vocab 文件在 load() 中加载，因为这种解码是有损的！
                s = render_token(token)
                # 查找这个标记的子标记，如果有的话
                if idx in inverted_merges:
                    # 如果这个标记有子标记，则将其渲染为合并形式
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则，这是一个叶子标记，直接打印
                    # （这应该是前256个标记，字节）
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        加载模型文件，与 save() 方法相反，但仅适用于 model 文件。
        
        参数:
            model_file (str): 需要加载的模型文件路径。
        
        异常:
            AssertionError: 如果文件不是 .model 文件或版本不匹配。
        """
        assert model_file.endswith(".model")
        # 读取模型文件
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # 读取版本信息
            version = f.readline().strip()
            assert version == "minbpe v1"
            # 读取模式字符串
            self.pattern = f.readline().strip()
            # 读取特殊标记的数量
            num_special = int(f.readline().strip())
            # 读取每个特殊标记及其对应的索引
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取合并规则
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        # 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出投影层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        # hidden_state: 输入的特征表示，形状为 [batch_size, seq_len, hidden_size]
        batch_size = hidden_state.size()[0]

        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # 将Q、K、V分为多个头
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_size))
        # 输出形状为[batch_size, num_heads, seq_len, seq_len]
        # 如果是掩码注意力，则加上掩码
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        # 对注意力分数进行归一化：softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_probs, value)
        # 输出形状为 [batch_size, num_heads, seq_len, head_size]

        # 对多个注意力进行拼接
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_size * self.num_heads)
        # [batch_size, seq_len, hidden_size]
        output = self.o_linear(output)

        return output

    def split_head(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size = x.size()[0]
        # 将hidden_size分为head_size * num_heads
        y = x.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        return y    # [batch_size, num_heads, seq_len, head_size]

if __name__ == "__main__":
    # 输入参数
    batch_size = 1
    seq_len = 4
    hidden_size = 8
    num_heads = 2

    # 构造固定输入
    input_tensor = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
         [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
         [0.3, 0.2, 0.1, 0.4, 0.7, 0.6, 0.5, 0.8],
         [0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5]]
    ])
    print(f"输入形状：{input_tensor.shape}")   # [1, 4, 8] [batch_size, seq_len, hidden_size]

    # 实例化注意力层
    mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    # 前向计算
    output = mha(input_tensor)
    # 输出结果
    print(f"输出形状：{output.shape}")  # [1, 4, 8]
    print(f"输出内容(保留四位小数)：\n {torch.round(output * 10000) / 10000}")

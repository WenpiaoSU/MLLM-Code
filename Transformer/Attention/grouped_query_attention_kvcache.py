import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_groups == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_groups = num_groups   # 组数
        self.head_dim = hidden_size // num_heads
        self.group_dim = self.head_dim * num_groups
        self.head_per_group = self.num_heads // self.num_groups   # 每组包含的Q头数
        # QKV线性层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.group_dim)
        self.v_linear = nn.Linear(hidden_size, self.group_dim)
        # 输出层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
                                   ).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
                                   ).cuda()

    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return x   # [batch_size, num_heads, seq_len, head_dim]
    def forward(self, x, mask = None):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x)   # [batch_size, seq_len, hidden_size]
        k = self.k_linear(x)   # [batch_size, seq_len, group_dim]
        v = self.v_linear(x)   # [batch_size, seq_len, group_dim]

        self.cache_k = self.cache_k.to(q.device)
        self.cache_v = self.cache_v.to(q.device)

        self.cache_k[batch_size, start_pos: start_pos + seq_len] = k
        self.cache_v[batch_size, start_pos: start_pos + seq_len] = v
        k = self.cache_k[batch_size, : start_pos + seq_len]
        v = self.cache_v[batch_size, : start_pos + seq_len]

        # 将Q分成多头
        q = self.split_heads(q)   # [batch_size, num_heads, seq_len, head_dim]
        # K,V分成多组
        # group_dim = head_dim * num_groups
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim)
        # 用广播(expand)将每组复制到该组对应的多个 heads
        # 得到 [B, L, num_groups, head_per_group, head_dim]
        k = k.unsqueeze(3).expand(-1, -1, -1, self.head_per_group, -1)
        v = v.unsqueeze(3).expand(-1, -1, -1, self.head_per_group, -1)
        # 把 group 与 head_per_group 合并，恢复成 [B, L, num_heads, head_dim]
        k = k.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 调整到 multi-head 常用布局 [B, num_heads, L, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, L, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, L, head_dim]

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e9'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        # 拼接 heads：先换回 [B, L, num_heads, head_dim]，再重排成 [B, L, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_linear(output)
        return output   # [batch_size, seq_len, hidden_size]

if __name__ == "__main__":
    # 输入参数
    batch_size = 1
    seq_len = 4
    hidden_size = 512
    num_heads = 8
    num_groups = 2

    # 构造随机输入
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    print(f"输入形状：{input_tensor.shape}")   # [1, 4, 512]

    # 实例化
    gqa = GroupedQueryAttention(hidden_size=hidden_size, num_heads=num_heads, num_groups=num_groups)
    # 前向计算
    output = gqa(input_tensor)
    # 输出结果
    print(f"输出形状：{output.shape}")  # [1, 4, 512]
    print(f"输出内容(保留四位小数)：\n {torch.round(output * 10000) / 10000}")




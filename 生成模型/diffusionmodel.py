# DDPM去噪扩散概率模型
import torch
import torch.nn as nn
import math

# 时间步编码 (Sinusoidal Time Embeddings)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        将时间步 t (batch_size) 转换为向量 (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2  # 一半维度用于sin，一半用于cos
        # 计算频率系数
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 时间步与频率相乘
        embeddings = time[:, None] * embeddings[None, :]
        # 拼接sin和cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# UNet 卷积块
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channel)
        if up:   # 上采样
            self.conv1 = nn.Conv2d(2*in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:    # 下采样
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv2d(out_channel, out_channel, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channel)
        self.bnorm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x, t):
        # 第一次卷积
        h = self.bnorm1(nn.ReLU(self.conv1(x)))
        # 注入时间编码
        time_emb = nn.ReLU(self.time_mlp(t))
        # 将时间编码广播到图像维度（Batch, Channel, 1, 1）
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb  # 特征融合
        # 第二次卷积
        h = self.bnorm2(nn.ReLU(self.conv2(h)))
        return self.transform(h)

# 简单的UNet模型
class SimpleUNet(nn.Module):
    """
    一个简化的 U-Net 结构，包含下采样路径和上采样路径，
    并在每一层注入时间信息。
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 3
        time_emb_dim = 32
        # 时间编码层
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        # 初始投影
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # 下采样
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        # 上采样
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # 1. 计算时间嵌入
        t = self.time_mlp(timestep)
        # 2. 初始卷积
        x = self.conv0(x)
        # 3. 下采样 (Encoder)
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        # 4. 上采样 (Decoder) + 跳跃连接 (Skip Connections)
        for up in self.ups:
            residual = residuals.pop()
            # 简单的拼接 (Concat)
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        return self.output(x)

# 扩散过程
class DiffusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # --- 预计算扩散参数 (Schedule) ---
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # alpha_hat (累乘 alpha)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        前向加噪过程: q(x_t | x_0)
        公式: x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        epsilon = torch.randn_like(x)  # 生成随机噪声
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """随机采样训练用的时间步"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, n):
        """
        反向去噪采样过程: p(x_{t-1} | x_t)
        从纯噪声开始，一步步去除噪声
        """
        model.eval()
        print(f"Sampling {n} new images....")

        # 1. 从标准正态分布采样纯噪声 x_T
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        with torch.no_grad():
            # 2. 从 T 到 1 倒序迭代
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)

                # 3. 预测噪声 (UNet 预测 epsilon)
                predicted_noise = model(x, t)

                # 4. 计算去噪参数
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)  # 最后一步不加噪声

                # 5. 去噪公式 (Langevin Dynamics)
                # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_hat) * predicted_noise) + sigma * z
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # 将像素值从 [-1, 1] 还原回 [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

# 测试运行
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化
    model = SimpleUNet().to(device)
    diffusion = DiffusionModel(img_size=64, device=device)

    # --- 模拟训练步骤 (Forward Process) ---
    print("--- 模拟训练输入 ---")
    batch_size = 4
    # 假设有一批真实图片
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    t = diffusion.sample_timesteps(batch_size)  # 随机采样时间步
    x_t, noise = diffusion.noise_images(images, t)  # 加噪得到 x_t
    predicted_noise = model(x_t, t)  # 模型尝试预测噪声

    # 计算 Loss (预测噪声和真实噪声的 MSE)
    loss = nn.MSELoss()(noise, predicted_noise)
    print(f"Training Loss: {loss.item()}")

    # --- 模拟生成步骤 (Reverse Process) ---
    print("\n--- 开始生成图片 (Reverse Sampling) ---")
    sampled_images = diffusion.sample(model, n=1)
    print(f"Generated Image Shape: {sampled_images.shape}")
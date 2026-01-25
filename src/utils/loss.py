import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        """
        InfoNCE Loss 实现 (基于 SimCLR 框架)
        :param batch_size: 批次大小
        :param temperature: 温度系数，越小模型对相似度的区分越敏感
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
        # 预先生成负样本掩码，避免每次 forward 都计算
        # 掩码的作用是：计算 loss 时，自己和自己不能比较
        self.mask = self.mask_correlated_samples(batch_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        # 生成一个 2N x 2N 的矩阵，把对角线置为 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        :param z_i: 第一个视图的特征向量 [Batch, Dim]
        :param z_j: 第二个视图的特征向量 [Batch, Dim]
        """
        N = 2 * self.batch_size
        
        # 1. 拼接两个视图的特征
        z = torch.cat((z_i, z_j), dim=0) # [2N, Dim]
        
        # 2. 计算余弦相似度矩阵
        # sim[i, j] 代表第 i 个样本和第 j 个样本的相似度
        sim = torch.matmul(z, z.T) / self.temperature
        
        # 3. 构造标签
        # 对于第 i 个样本，它的正样本是第 (i + batch_size) 个
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # 将正样本对的相似度拼在一起
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # 负样本是除正样本以外的所有样本
        # 这里为了简化，我们使用一个标准的 CrossEntropy 实现 trick
        # 实际实现中，SimCLR 的 loss 计算稍微复杂一点，
        # 为了你的论文代码易读性，我们采用更直观的 NT-Xent 实现方式：
        
        # 重新计算标准的 NT-Xent Loss
        # 归一化特征
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 过滤掉自己与自己的相似度
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix),
            1,
            torch.arange(N).view(-1, 1).to(self.device),
            0
        )
        
        # 只要非对角线部分的相似度
        mask = torch.eye(N, dtype=torch.bool).to(self.device)
        # 这里的 label 是：对于第 k 个样本，哪个索引是它的正样本？
        # 如果 k < batch_size, 正样本是 k + batch_size
        # 如果 k >= batch_size, 正样本是 k - batch_size
        
        # === 极简版实现 (推荐) ===
        # 上面的逻辑是为了讲清楚原理，下面是 PyTorch 社区通用的极简写法
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels + self.batch_size - 1) % (2 * self.batch_size) # 这是一个错位 label
        
        # 实际上，最简单的 InfoNCE 只要保证：
        # 分子：exp(sim(z_i, z_j) / temp)
        # 分母：sum(exp(sim(z_i, z_k) / temp))
        
        # 让我们用一个最稳健的库函数风格：
        # Cosine similarity between z_i and z_j
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Labels: 0->N, 1->N+1 ... N->0
        # 这是一个分类问题：在 2N-1 个候选中找到唯一的那个正样本
        
        # 构造 target
        labels = torch.cat([torch.arange(self.batch_size) + self.batch_size, torch.arange(self.batch_size)], dim=0).to(self.device)
        
        # 屏蔽对角线 (自己和自己)
        mask = torch.eye(N, dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15) # 负无穷
        
        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * self.batch_size)
"""精简版 Multi-Agent Transformer（离散动作），支持四种出动作顺序。

order_mode:
  fixed   按仓库编号 0..N-1 依次出动作（原版 MAT）
  random  每一步随机打乱顺序（HAPPO / MAT 论文中的随机顺序）
  learned 打分网络 + Plackett-Luce 采样学习顺序（PMAT 的核心思想）
  none    不做自回归，每个智能体只看编码器表示（相当于共享参数的 MAPPO 式并行决策）

约定：perm[b, k] = 第 k 个出动作的智能体编号。所有对外接口的张量都按智能体编号排列，
只有解码器内部按 perm 排列。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ORDER_MODES = ("fixed", "random", "learned", "none")


def init_(m, gain=0.01, relu=False):
    if relu:
        gain = nn.init.calculate_gain("relu")
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, causal):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.causal = n_head, causal
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))

    def forward(self, key, value, query):
        B, L, D = query.shape
        h = self.n_head
        k = self.key(key).view(B, L, h, D // h).transpose(1, 2)
        q = self.query(query).view(B, L, h, D // h).transpose(1, 2)
        v = self.value(value).view(B, L, h, D // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if self.causal:
            mask = torch.ones(L, L, dtype=torch.bool, device=att.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, causal=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        return self.ln2(x + self.mlp(x))


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(n_embd) for _ in range(3))
        self.attn1 = SelfAttention(n_embd, n_head, causal=True)
        self.attn2 = SelfAttention(n_embd, n_head, causal=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x, rep):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep + self.attn2(key=x, value=x, query=rep))
        return self.ln3(x + self.mlp(x))


def gather_agents(x, perm):
    """x: (B, N, ...) 按 perm 重排成决策顺序。"""
    idx = perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand(*perm.shape, *x.shape[2:])
    return torch.gather(x, 1, idx)


def scatter_agents(x, perm):
    """gather_agents 的逆操作：决策顺序 -> 智能体编号顺序。"""
    out = torch.empty_like(x)
    idx = perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand(*perm.shape, *x.shape[2:])
    return out.scatter_(1, idx, x)


def plackett_luce_log_prob(scores, perm):
    """scores: (B, N)，perm: (B, N)。返回排列的对数概率 (B,)。"""
    s = torch.gather(scores, 1, perm)
    # 第 k 位的归一化项是剩余元素（k..N-1）的 logsumexp
    tail_lse = torch.logcumsumexp(s.flip(1), dim=1).flip(1)
    return (s - tail_lse).sum(dim=1)


def plackett_luce_entropy_proxy(scores):
    """首位选择分布的熵，用作顺序探索的正则（精确熵需要枚举 N! 个排列）。"""
    return Categorical(logits=scores).entropy()


class MultiAgentTransformer(nn.Module):
    def __init__(self, obs_dim, n_actions, n_agents, n_embd=64, n_head=2, n_block=1,
                 order_mode="fixed"):
        super().__init__()
        if order_mode not in ORDER_MODES:
            raise ValueError(f"order_mode 必须是 {ORDER_MODES} 之一")
        self.n_agents, self.n_actions, self.order_mode = n_agents, n_actions, order_mode

        # 编码器：观测 -> 表示 + 每个智能体的价值
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), relu=True), nn.GELU())
        self.enc_ln = nn.LayerNorm(n_embd)
        self.enc_blocks = nn.ModuleList(EncodeBlock(n_embd, n_head) for _ in range(n_block))
        self.value_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                        nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, 1)))

        # 解码器：上一个智能体的动作（one-hot，额外一位作起始符）-> 当前智能体的动作分布
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(n_actions + 1, n_embd, bias=False), relu=True), nn.GELU())
        self.dec_ln = nn.LayerNorm(n_embd)
        self.dec_blocks = nn.ModuleList(DecodeBlock(n_embd, n_head) for _ in range(n_block))
        self.policy_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                         nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, n_actions)))

        # 顺序打分网络（只在 learned 模式下使用）
        self.order_scorer = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                          init_(nn.Linear(n_embd, 1)))

    # ------------------------------------------------------------------ #
    def encode(self, obs):
        x = self.enc_ln(self.obs_encoder(obs))
        for blk in self.enc_blocks:
            x = blk(x)
        return x, self.value_head(x).squeeze(-1)

    def order_scores(self, rep):
        return self.order_scorer(rep).squeeze(-1)

    def sample_order(self, rep, deterministic=False):
        """返回 perm (B, N) 和它的对数概率 (B,)；非 learned 模式的对数概率恒为 0。"""
        B, N = rep.shape[:2]
        dev = rep.device
        if self.order_mode == "learned":
            scores = self.order_scores(rep)
            if deterministic:
                perm = scores.argsort(dim=1, descending=True)
            else:  # Gumbel-top-k 技巧等价于 Plackett-Luce 采样
                g = -torch.log(-torch.log(torch.rand_like(scores).clamp_min(1e-10)))
                perm = (scores + g).argsort(dim=1, descending=True)
            return perm, plackett_luce_log_prob(scores, perm)
        if self.order_mode == "random":  # 随机顺序在评估时也保持随机，deterministic 只影响动作
            perm = torch.rand(B, N, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(N, device=dev).expand(B, N)
        return perm, torch.zeros(B, device=dev)

    def _decode(self, shifted_onehot, rep_ordered):
        x = self.dec_ln(self.action_encoder(shifted_onehot))
        for blk in self.dec_blocks:
            x = blk(x, rep_ordered)
        return self.policy_head(x)

    def _start_token(self, B, N, dev):
        tok = torch.zeros(B, N, self.n_actions + 1, device=dev)
        tok[..., -1] = 1.0
        return tok

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs: (B, N, obs_dim)。返回 actions, logp(每个智能体), values, perm, order_logp。"""
        B, N = obs.shape[:2]
        dev = obs.device
        rep, values = self.encode(obs)
        perm, order_logp = self.sample_order(rep, deterministic)
        rep_o = gather_agents(rep, perm)

        dec_in = self._start_token(B, N, dev)
        actions_o = torch.zeros(B, N, dtype=torch.long, device=dev)
        logp_o = torch.zeros(B, N, device=dev)

        if self.order_mode == "none":
            dist = Categorical(logits=self._decode(dec_in, rep_o))
            actions_o = dist.probs.argmax(-1) if deterministic else dist.sample()
            logp_o = dist.log_prob(actions_o)
        else:
            for k in range(N):  # 自回归：第 k 个智能体看得到前 k-1 个的动作
                logits = self._decode(dec_in, rep_o)[:, k]
                dist = Categorical(logits=logits)
                a = logits.argmax(-1) if deterministic else dist.sample()
                actions_o[:, k], logp_o[:, k] = a, dist.log_prob(a)
                if k + 1 < N:
                    dec_in[:, k + 1] = F.one_hot(a, self.n_actions + 1).float()

        actions = scatter_agents(actions_o, perm)
        logp = scatter_agents(logp_o, perm)
        return actions, logp, values, perm, order_logp

    def evaluate(self, obs, actions, perm):
        """训练时用 teacher forcing 一次性算出所有智能体的 logp 和熵。"""
        B, N = obs.shape[:2]
        rep, values = self.encode(obs)
        rep_o = gather_agents(rep, perm)
        dec_in = self._start_token(B, N, obs.device)
        if self.order_mode != "none":
            actions_o = gather_agents(actions, perm)
            dec_in[:, 1:] = F.one_hot(actions_o[:, :-1], self.n_actions + 1).float()
        dist = Categorical(logits=self._decode(dec_in, rep_o))
        logp = scatter_agents(dist.log_prob(gather_agents(actions, perm)), perm)
        entropy = scatter_agents(dist.entropy(), perm)

        if self.order_mode == "learned":
            scores = self.order_scores(rep)
            order_logp = plackett_luce_log_prob(scores, perm)
            order_entropy = plackett_luce_entropy_proxy(scores)
        else:
            order_logp = order_entropy = torch.zeros(B, device=obs.device)
        return logp, entropy, values, order_logp, order_entropy

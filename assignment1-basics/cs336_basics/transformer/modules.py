from re import T
import torch
import torch.nn as nn
from math import sqrt
from einops import einsum,reduce,rearrange
from jaxtyping import Float, Int
import math
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
import random


# 1 point 
class Linear(torch.nn.Module):
    """
    Note: The bias is missing.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, 'batch ... input, output input -> batch ... output')
    
# 1 point
class Embeeding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]  # shape: [batch, seq_len, embedding_dim]

# 1 point
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype=None):
        super().__init__()
        self.d_mod = d_model
        self.eps = eps
        self.device = device
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        # save type before rescaline due to the possible overflow after squaring the input 
        in_t = input_tensor.dtype
        input_tensor = input_tensor.to(dtype=torch.float32)
        rms = torch.sqrt((input_tensor.pow(2).mean(dim=-1, keepdim=True)) + self.eps)
        out = (input_tensor / rms) * self.weight 
        return out.to(dtype=in_t, device=self.weight.device)
        



class SwigLUFeedForward(torch.nn.Module):
    def __init__(self, d_model : int, d_ff : int = None, device = None, dtype = None):
        super().__init__()
        if d_ff is None:
            self.d_ff = int(8 / 3 * d_model) # 8/3 of our dmodel, ensuring it is multiplebale by 64 cause gpus 
            self.d_ff = (self.d_ff + 63) // 64 * 64 # (n + m-1) // m * m famous trick
        else:
            self.d_ff = d_ff
        self.weights1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.weights2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.weights3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x : torch.Tensor):

        first_forward = self.weights1(x) # d_model -> d_ff
        second_forward = self.weights3(x) # d_model -> d_ff
        silu = first_forward * torch.sigmoid(first_forward) # swish
        swiglu = second_forward * silu 
        return self.weights2(swiglu) # output is d_model 



class RotatoryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta : float, d_k : int, max_seq_len : int, device=None):
        super().__init__()
        self.theta = theta # constant 
        self.d_k = d_k # dimension of key and query values 
        self.max_len = max_seq_len
        self.device = device

        pair_index = d_k // 2
        # intializing our computational matrix
        k_index = torch.arange(pair_index, device=self.device).repeat_interleave(2)
        freq_matrix = 1 / theta ** (2 * k_index / d_k) # [1, num_dim / 2]
        max_seq = torch.arange(0, max_seq_len, device=self.device) 
        # the rotation matrix tells us by how much do we want to rotate every dimension, 
        rotation_matrix = einsum(freq_matrix, max_seq, "dk, mcl -> dk mcl")
        cos_matrix = torch.cos(rotation_matrix)
        sin_matrix = torch.sin(rotation_matrix)

        self.register_buffer("cos_matrix",cos_matrix, persistent=False)
        self.register_buffer("sin_matrix",sin_matrix, persistent=False)
        
    def forward(self, x : torch.Tensor, token_positions = None) -> torch.Tensor:
        # cos matrix is shape [emb_dim, max_context_len, we create splice based on token positions]
        # (..., seq_len, d_k) x shape 
        
        cos = self.cos_matrix[:, :x.size(-2)].to(x.dtype)
        sine = self.sin_matrix[:, :x.size(-2)].to(x.dtype) # casting 
        x_cos = x * cos.T.unsqueeze(0)
        y = x.clone()
        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] =  x[..., 0::2]
        y_sin = y * sine.T.unsqueeze(0)
        a = x_cos + y_sin
       
        return a






def softmax(in_features : torch.Tensor, dim : int) -> torch.Tensor:
    max_val_vec = torch.max(in_features, dim=dim, keepdim=True).values
    norm_in = torch.exp(in_features - max_val_vec)

    sum = torch.sum(norm_in, dim=dim, keepdim=True)
    return norm_in / sum

def cross_entropy(inputs : Float[torch.Tensor, "batch_size vocab_size"], targets : Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:

    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1).long())
    lem = torch.logsumexp(inputs, -1, keepdim=True)
    loss = -target_logits + lem
    
    return loss.mean()
def run_scaled_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    matmul = (Q @ K.transpose(-2,-1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        matmul = matmul.masked_fill(~mask.bool(), float("-inf"))
    return softmax(matmul, -1) @ V


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model : int, num_heads: int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_qkv = d_model // num_heads
        self.device = device 
        self.dtype = dtype

        assert d_model % num_heads == 0 ," Wrong num of heads"

        self.qweight = Linear(d_model, d_model, device = device, dtype=dtype) # a nn, which projects for dim_qkv outputs for n_heads heads
        self.kweight = Linear(d_model, d_model, device = device, dtype=dtype)
        self.vweight = Linear(d_model, d_model, device = device, dtype=dtype)
        self.oweight = Linear(d_model, d_model, device = device, dtype=dtype)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            x: (batch, ..., seq_len, d_model)
        Returns:
            (batch, ..., seq_len, d_model)
        """

        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]
        
        Q = self.qweight(x)
        K = self.kweight(x)
        V = self.vweight(x)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn_out = run_scaled_attention(Q, K, V, mask=causal_mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.oweight(attn_out)

class MultiheadSelfAttentionRope(MultiheadSelfAttention):
    def __init__(self, d_model: int, num_heads: int, theta: int, max_seq_len: int, device=None, dtype=None):
        super().__init__(d_model, num_heads, device = device,dtype=dtype)
        self.device = device
        self.RoPE = RotatoryPositionalEmbedding(theta, self.dim_qkv, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)
        seq_len = x.shape[-2]
        Q = self.qweight(x)
        K = self.kweight(x)
        V = self.vweight(x)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads, d=self.dim_qkv)
        Q = self.RoPE(Q)
        K = self.RoPE(K)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attn_out = run_scaled_attention(Q, K, V, mask=causal_mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.oweight(attn_out)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ff : int, max_seq_len : int,theta = 10000, device = None, dtype = None,
                 use_rope = True, use_swiglu = True):
        # our transformer block contains of two RMS Norms and One attention with rope, and one SwiGLU
        super().__init__()
        self.d_model = d_model # emb dimmension
        self.num_heads = num_heads
        self.dim_kqv = d_model // num_heads
        self.max_seq_len = max_seq_len
        assert d_model % num_heads == 0, "Number of heads need to be the divident of embeeding dimension"
        self.device = device
        self.dtype = dtype
        self.RMSNorm1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.RMSNorm2 = RMSNorm(d_model, device = device, dtype = dtype)
        if use_rope:
            self.MulAttentionRope = MultiheadSelfAttentionRope(d_model, num_heads, theta, max_seq_len, device = device, dtype = dtype)
        else:
            self.MulAttentionRope = MultiheadSelfAttention(d_model, num_heads, device = device, dtype = dtype)
        if use_swiglu:
            self.SwigLU = SwigLUFeedForward(d_model, d_ff, device = device, dtype = dtype)
        else:
            self.SwigLU = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
    def forward(self, x: torch.Tensor):
        
        residual1 = x
        out1 = self.RMSNorm1(x)
        out1 = self.MulAttentionRope(out1)
        first_pass = out1 + residual1.to(out1.device)
        
        residual2 = first_pass
        out2 = self.RMSNorm2(first_pass)
        out2 = self.SwigLU(out2)
        second_pass = out2 + residual2.to(out2.device)
        
        return second_pass


class TransformerLM(torch.nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int = None,
                 rope_theta: float = 10000.0,
                 device=None,
                 dtype=None,
                 use_rope: bool = True,
                 use_swiglu: bool = True,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_con = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.token_emb = Embeeding(vocab_size, d_model, device=device)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, 
            theta=rope_theta, device=device, dtype=dtype, use_rope=use_rope, use_swiglu=use_swiglu)
            for _ in range(num_layers)
        ])
        self.final_rms_norm = RMSNorm(d_model, device = device, dtype = dtype)
        self.out_emb = Linear(d_model, vocab_size, device=device, dtype=dtype) 
    def forward(self, x):
        x = self.token_emb(x)
        for b in self.blocks:
            x = b(x)
        x = self.final_rms_norm(x)
        x = self.out_emb(x)
        return x


class SGDOptimizerDecaying(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0:
            raise ValueError(f'Invalid learning rate {lr}, cannot be lower than zero')
        defaults = {"lr" : lr}
        super().__init__(params, defaults) # we pass to superclass
    
    def step(self, closure: Optional[Callable] = None):
        # closure is our loss
        loss = None if closure is None else closure()
        for g in self.param_groups: # for each layer, grab their params 
            lr = g["lr"]
            for p in g["params"]:
                if p.grad is None:
                    continue # why? ask?
                state = self.state[p]
                t = state.get("t", 0) # the number of iteration
                grad = p.grad.data # gradient respect to the loss
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), weight_decay = 0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f'Invalid learning rate {lr}, cannot be lower than zero')
        defaults = {
            "lr" : lr,
            "beta1" : betas[0],
            "beta2" : betas[1],
            "w_dec" : weight_decay,
            "eps" : eps,
        }
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for g in self.param_groups: # for each layer grab their params
            lr = g["lr"]
            beta1 = g["beta1"]
            beta2 = g["beta2"]
            w_dec = g["w_dec"]
            eps = g["eps"]
            for p in g["params"]: # for each parametr in the layer 
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1) # we start from one 
                if t == 1:
                    m = torch.zeros_like(p.data)
                    v = torch.zeros_like(p.data)
                    state["t"] = t
                else:
                    m = state["m"]
                    v = state["v"]
                grad = p.grad.data # gradient respect to the loss

                state["m"] = m * beta1 + (1 - beta1) * grad
                state["v"] = v * beta2 + (1 - beta2) * (grad ** 2)

                real_lr = lr * ((math.sqrt(1 - (beta2 ** t))) / (1 - (beta1 ** t)))
                p.data -= real_lr * ( (state["m"]) / (torch.sqrt(state["v"]) + eps))
                p.data -= lr * w_dec * p.data
                state["t"] += 1
            
        return loss


def learning_rate_schedule(t, lrmax, lrmin, t_warmup_c, t_cosine_c):
    if lrmax < 0 or lrmin < 0:
        raise ValueError("Learning rate cannot be negative number")
    if t < t_warmup_c:
        return t/t_warmup_c * lrmax

    elif t_warmup_c <= t <= t_cosine_c:

        return lrmin + 0.5 * (1 + math.cos((math.pi * (t - t_warmup_c)) / (t_cosine_c - t_warmup_c))) * (lrmax - lrmin)
    elif t > t_cosine_c:
        return lrmin
    else:
        raise Exception("Something went wrong with the scheduler")

def gradient_clipping(params, M, eps=1e-6):
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    norm = total_sq ** 0.5
    clip_coef = 1.0 if norm <= M else M / (norm + eps)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return norm
            

def data_get_batch(
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = dataset.shape[0] - context_length - 1
    rand_starts = np.random.randint(0, max_start, size=batch_size)

    
    idx = rand_starts[:, None] + np.arange(context_length)  

    
    batch_inputs = dataset[idx]
    batch_labels = dataset[idx + 1]

    
    input_tensor = torch.from_numpy(batch_inputs).to(device=device, dtype=dtype, non_blocking=True)
    label_tensor = torch.from_numpy(batch_labels).to(device=device, dtype=dtype, non_blocking=True)
    return input_tensor, label_tensor



def save_checkpoint(model, optimizer, iteration, out):
    return torch.save({
        "m_state_dict": model.state_dict(),
        "o_state_dict": optimizer.state_dict(),
        "i": iteration,
    }, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["m_state_dict"])
    optimizer.load_state_dict(checkpoint["o_state_dict"])
    return checkpoint["i"]





import math
from functools import wraps, partial
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, einsum

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

from audiolm_pytorch import AudioLM
from audiolm_pytorch.utils import AudioConditionerBase

import torch.distributed as dist
from musiclm_pytorch.distributed import AllGather

from x_clip.tokenizer import tokenizer
from vector_quantize_pytorch import ResidualVQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype.typing import List, Optional, Tuple
from beartype import beartype

# functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(val, d):
    return val if exists(val) else d

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# tensor functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# biasless layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        layers = []

        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x
            layers.append(x)

        if not return_all_layers:
            return x

        return x, torch.stack(layers[:-1])

# contrastive losses

class SoftmaxContrastiveLearning(nn.Module):
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp()

        cosine_sims_exp = sims.exp()

        numerator = matrix_diag(cosine_sims_exp)

        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()

class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        device = self.device

        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        text_latents, rank_sizes = self.all_gather(text_latents)

        n = text_latents.shape[1]

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp() + self.bias

        labels = torch.eye(n, device = device)

        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0)
            labels = labels_by_ranks[dist.get_rank()]

        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        return -F.logsigmoid(labels * sims).sum() / n

# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 2048,  # Optimized for better frequency resolution
        spec_power = 2,
        spec_win_length = 512,  # Optimized window length
        spec_hop_length = None,  # Will be calculated as win_length // 4
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 27,  # Optimized frequency masking
        spec_aug_time_mask = 100,  # Optimized time masking
        patch_dropout_prob = 0.25,
        support_variable_length = True,  # New parameter for variable length support
        max_audio_length = 1024 * 320  # Maximum audio length in samples (about 20 seconds at 16kHz)
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first
        self.support_variable_length = support_variable_length
        self.max_audio_length = max_audio_length
        
        # Calculate hop_length if not provided for optimal overlap
        if spec_hop_length is None:
            spec_hop_length = spec_win_length // 4
            
        self.spec_hop_length = spec_hop_length

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = self.spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779

        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )

        self.norm = LayerNorm(dim)

        # patch dropout

        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias

        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False,
        return_all_layers = False,
        audio_lengths = None  # New parameter for variable length audio support
    ):
        batch, device = x.shape[0], x.device
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        # Handle variable length audio with masking
        if self.support_variable_length and audio_lengths is not None:
            # Create attention mask for variable length sequences
            if not self.accept_spec:
                # Calculate expected spectrogram dimensions
                hop_length = self.spec_hop_length
                n_fft = self.spec.n_fft
                expected_time_frames = (audio_lengths + hop_length - 1) // hop_length
                max_time_frames = (self.max_audio_length + hop_length - 1) // hop_length
                
                # Create padding mask
                mask = torch.arange(max_time_frames, device=device).unsqueeze(0) < expected_time_frames.unsqueeze(1)
            else:
                # For pre-computed spectrograms, use the provided lengths directly
                mask = audio_lengths
        else:
            mask = None

        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            x = self.spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # to patches

        x = self.to_patch_tokens(x)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions

        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding

        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout

        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias

        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else

        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, mask = mask, return_all_layers = True)

        # final global average and norm (most recent papers show this is superior to CLS token)

        x = reduce(x, 'b n d -> b d', 'mean')

        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers

# text transformer

class TextTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.depth = depth
        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None,
        return_all_layers = False
    ):
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device

        # token embedding + positional embedding

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value = True)

        # attention

        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)

        # unpack the cls tokens

        cls_tokens, _ = unpack(x, ps, 'b * d')

        out = self.norm(cls_tokens)

        if not return_all_layers:
            return out

        return out, all_layers

# hierarchical cl loss

def interspersed_indices(layers, total_layers):
    assert total_layers >= layers
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()

class MultiLayerContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        audio_dim,
        text_dim,
        dim_latent,
        layers,
        decoupled_contrastive_learning = False,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.layers = layers

        self.audio_norm = LayerNorm(audio_dim, scale = False)
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        self.text_norm = LayerNorm(text_dim, scale = False)
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        self.contrast = klass(layers = layers)

    def forward(self, *, audio_layers, text_layers):
        device, batch = audio_layers.device, audio_layers.shape[1]

        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        audio_latents = l2norm(audio_latents)

        text_cls_tokens = text_layers[:, :, 0]
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        text_latents = l2norm(text_latents)

        return self.contrast(audio_latents, text_latents)

# main classes

class MuLaN(nn.Module):
    @beartype
    def __init__(
        self,
        audio_transformer: AudioSpectrogramTransformer,
        text_transformer: TextTransformer,
        dim_latent = 128,                       # they use 128
        decoupled_contrastive_learning = True,  # think this was used, make it optional
        hierarchical_contrastive_loss = False,
        hierarchical_contrastive_loss_layers = None,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.audio = audio_transformer
        self.text = text_transformer


        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        self.contrast = klass()

        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    def get_audio_latents(
        self,
        wavs,
        return_all_layers = False
    ):
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        audio_latents = self.audio_to_latents(audio_embeds)
        out = l2norm(audio_latents)

        if not return_all_layers:
            return out

        return out, audio_layers

    @beartype
    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_all_layers = False
    ):
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        text_latents = self.text_to_latents(text_embeds)
        out = l2norm(text_latents)

        if not return_all_layers:
            return out

        return out, text_layers

    @beartype
    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False
    ):
        batch, device = wavs.shape[0], wavs.device

        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers = True)
        text_latents, text_layers = self.get_text_latents(texts, raw_texts = raw_texts, return_all_layers = True)

        if return_latents:
            return audio_latents, text_latents

        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim

        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        # whether to do cl loss across all layers, from ViCHA paper https://arxiv.org/abs/2208.13628

        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers,
            text_layers = text_layers
        )

        return cl_loss + hierarchical_cl_loss

# music lm

class MuLaNEmbedQuantizer(AudioConditionerBase):
    @beartype
    def __init__(
        self,
        mulan: MuLaN,
        conditioning_dims: Tuple[int, ...],
        rq_num_quantizers = 8,
        rq_ema_decay = 0.9,
        codebook_size = 1024,
        namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),

    ):
        super().__init__()
        self.mulan = mulan

        assert len(namespaces) > 0
        self.namespaces = namespaces
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(namespaces), 'number of conditioning dimensions must be equal to number of namespaces'

        dim = mulan.dim_latent

        self.rq = ResidualVQ(
            dim = dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = 0,    # only use EMA to update codebooks
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = False  # no quantize dropout
        )

        self.dim = dim
        self.num_codebooks = rq_num_quantizers

        self.cond_embeddings = nn.ParameterDict({})

        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            nn.init.normal_(cond_embeddings, std = 0.02)

            self.cond_embeddings[namespace] = cond_embeddings

        self.set_default_namespace(namespaces[0])

    def parameters(self):
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    def forward(
        self,
        wavs = None,
        texts = None,
        namespace = None
    ):
        assert exists(wavs) ^ exists(texts)

        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f'namespace {namespace} not found'
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.mulan.eval()

            # sound and language live in joint embedding space because of contrastive learning

            if exists(wavs):
                latents = self.mulan.get_audio_latents(wavs)
            elif exists(texts):
                latents = self.mulan.get_text_latents(texts)

        _, indices, _ = self.rq(latents)

        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]

        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b = batch)
        indices = repeat(indices, 'b q -> b q 1 d', q = num_codebooks, d = dim)

        cond_embeddings = cond_embeddings.gather(2, indices)
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')

class MusicLM(nn.Module):
    @beartype
    def __init__(
        self,
        audio_lm: AudioLM,
        mulan_embed_quantizer: MuLaNEmbedQuantizer
    ):
        super().__init__()
        assert not exists(audio_lm.audio_conditioner), 'mulan must not have been passed into AudioLM. it will be managed externally now, embedding the text into the joint embedding space for text-to-audio synthesis'

        self.mulan_embed_quantizer = mulan_embed_quantizer
        self.audio_lm = audio_lm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        text: str,
        num_samples = 1,
        **audio_lm_kwargs
    ):
        self.eval()

        texts = tokenizer.tokenize([text]).to(self.device)

        text_embeds = self.mulan_embed_quantizer(texts = texts)

        # unable to deal with variable lengthed audio for now

        samples = []

        for _ in range(num_samples):
            music = self.audio_lm(text_embeds = text_embeds, **audio_lm_kwargs)
            samples.append(music)

        # if one sample, just return it

        if num_samples == 1:
            return first(samples)

        mulan = self.mulan_embed_quantizer.mulan

        # get the one with the highest similarity score, of all the samples

        sims = torch.cat([mulan(texts = texts, wavs = music, return_similarities = True) for music in samples], dim = 0)
        top_matching_index = sims.topk(1, dim = 0).indices.item()

        return samples[top_matching_index]


# OpenCLIP Integration
# MuLaN adapter for open_clip compatibility

class MuLaNOpenCLIPAdapter(nn.Module):
    """
    Adapter to use MuLaN with OpenCLIP framework.
    This allows MuLaN to be used as an audio-text contrastive model
    within the OpenCLIP ecosystem.
    """
    
    def __init__(
        self,
        mulan: MuLaN,
        audio_input_dim: int = 1024,
        text_input_dim: int = 512,
        embed_dim: int = 512,
        audio_patch_size: int = 16,
        text_patch_size: int = 1
    ):
        super().__init__()
        self.mulan = mulan
        self.embed_dim = embed_dim
        
        # Projection layers to match OpenCLIP dimensions
        self.audio_projection = nn.Linear(mulan.audio.dim, embed_dim)
        self.text_projection = nn.Linear(mulan.text.dim, embed_dim)
        
        # Store patch sizes for OpenCLIP compatibility
        self.audio_patch_size = audio_patch_size
        self.text_patch_size = text_patch_size
        
        # Vision model-like interface for OpenCLIP
        self.visual = mulan.audio  # Audio acts as "visual" modality
        self.text_encoder = mulan.text
        
        # Logit scale for contrastive learning (OpenCLIP style)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_audio(self, audio, audio_lengths=None, normalize=True):
        """Encode audio using MuLaN audio transformer."""
        # Get audio embeddings from MuLaN
        if audio_lengths is not None:
            audio_features = self.mulan.audio(audio, audio_lengths=audio_lengths)
        else:
            audio_features = self.mulan.audio(audio)
            
        # Project to OpenCLIP embedding space
        audio_embeddings = self.audio_projection(audio_features)
        
        if normalize:
            audio_embeddings = F.normalize(audio_embeddings, dim=-1)
            
        return audio_embeddings
    
    def encode_text(self, text, normalize=True):
        """Encode text using MuLaN text transformer."""
        # Get text embeddings from MuLaN
        text_features = self.mulan.text(text)
        
        # Project to OpenCLIP embedding space
        text_embeddings = self.text_projection(text_features)
        
        if normalize:
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            
        return text_embeddings
    
    def forward(self, audio, text, audio_lengths=None):
        """
        Forward pass for contrastive learning.
        Returns audio and text embeddings for contrastive loss computation.
        """
        audio_embeddings = self.encode_audio(audio, audio_lengths)
        text_embeddings = self.encode_text(text)
        
        # Get logit scale
        logit_scale = self.logit_scale.exp()
        
        return audio_embeddings, text_embeddings, logit_scale
    
    def get_similarity(self, audio, text, audio_lengths=None):
        """Get similarity scores between audio and text pairs."""
        audio_embeddings, text_embeddings, logit_scale = self.forward(audio, text, audio_lengths)
        
        # Compute similarity matrix
        logits_per_audio = logit_scale * audio_embeddings @ text_embeddings.t()
        logits_per_text = logit_scale * text_embeddings @ audio_embeddings.t()
        
        return logits_per_audio, logits_per_text


def create_mulan_open_clip_model(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    embed_dim=512,
    audio_sample_rate=16000,
    use_case='music',
    spec_n_fft=None,  # Will be set by get_optimal_spectrogram_params
    spec_win_length=None,  # Will be set by get_optimal_spectrogram_params
    spec_hop_length=None,  # Will be set by get_optimal_spectrogram_params
    decoupled_contrastive_learning=True,
    sigmoid_contrastive_loss=False
):
    """
    Factory function to create a MuLaN model adapted for OpenCLIP.
    
    Args:
        dim: Model dimension
        depth: Transformer depth
        heads: Number of attention heads
        dim_head: Dimension per head
        embed_dim: Embedding dimension for OpenCLIP compatibility
        audio_sample_rate: Audio sample rate
        use_case: One of 'music', 'speech', 'general'
        spec_n_fft: FFT size for spectrogram (if None, uses optimal params)
        spec_win_length: Window length for spectrogram (if None, uses optimal params)
        spec_hop_length: Hop length for spectrogram (if None, uses optimal params)
        decoupled_contrastive_learning: Use decoupled contrastive learning
        sigmoid_contrastive_loss: Use sigmoid contrastive loss
    
    Returns:
        MuLaNOpenCLIPAdapter instance ready for OpenCLIP training
    """
    
    # Get optimal spectrogram parameters if not specified
    if spec_n_fft is None or spec_win_length is None:
        spec_params = get_optimal_spectrogram_params(audio_sample_rate, use_case)
        spec_n_fft = spec_n_fft or spec_params['spec_n_fft']
        spec_win_length = spec_win_length or spec_params['spec_win_length']
        spec_hop_length = spec_hop_length or spec_params.get('spec_hop_length')
    
    # Create audio transformer with optimized parameters
    audio_transformer = AudioSpectrogramTransformer(
        dim=dim,
        depth=depth,
        patch_size=16,
        dim_head=dim_head,
        heads=heads,
        spec_n_fft=spec_n_fft,
        spec_win_length=spec_win_length,
        spec_hop_length=spec_hop_length,
        spec_aug_freq_mask=27,  # Optimized for music
        spec_aug_time_mask=100,
        support_variable_length=True,
        max_audio_length=audio_sample_rate * 20  # 20 seconds max
    )
    
    # Create text transformer
    text_transformer = TextTransformer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head
    )
    
    # Create MuLaN model
    mulan = MuLaN(
        audio_transformer=audio_transformer,
        text_transformer=text_transformer,
        dim_latent=embed_dim,
        decoupled_contrastive_learning=decoupled_contrastive_learning,
        sigmoid_contrastive_loss=sigmoid_contrastive_loss
    )
    
    # Wrap in OpenCLIP adapter
    return MuLaNOpenCLIPAdapter(
        mulan=mulan,
        audio_input_dim=dim,
        text_input_dim=dim,
        embed_dim=embed_dim
    )
    """
    Factory function to create a MuLaN model adapted for OpenCLIP.
    
    Args:
        dim: Model dimension
        depth: Transformer depth
        heads: Number of attention heads
        dim_head: Dimension per head
        embed_dim: Embedding dimension for OpenCLIP compatibility
        audio_sample_rate: Audio sample rate
        spec_n_fft: FFT size for spectrogram
        spec_win_length: Window length for spectrogram
        spec_hop_length: Hop length for spectrogram
        decoupled_contrastive_learning: Use decoupled contrastive learning
        sigmoid_contrastive_loss: Use sigmoid contrastive loss
    
    Returns:
        MuLaNOpenCLIPAdapter instance ready for OpenCLIP training
    """
    
    # Create audio transformer with optimized parameters
    audio_transformer = AudioSpectrogramTransformer(
        dim=dim,
        depth=depth,
        patch_size=16,
        dim_head=dim_head,
        heads=heads,
        spec_n_fft=spec_n_fft,
        spec_win_length=spec_win_length,
        spec_hop_length=spec_hop_length,
        spec_aug_freq_mask=27,  # Optimized for music
        spec_aug_time_mask=100,
        support_variable_length=True,
        max_audio_length=audio_sample_rate * 20  # 20 seconds max
    )
    
    # Create text transformer
    text_transformer = TextTransformer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head
    )
    
    # Create MuLaN model
    mulan = MuLaN(
        audio_transformer=audio_transformer,
        text_transformer=text_transformer,
        dim_latent=embed_dim,
        decoupled_contrastive_learning=decoupled_contrastive_learning,
        sigmoid_contrastive_loss=sigmoid_contrastive_loss
    )
    
    # Wrap in OpenCLIP adapter
    return MuLaNOpenCLIPAdapter(
        mulan=mulan,
        audio_input_dim=dim,
        text_input_dim=dim,
        embed_dim=embed_dim
    )


def get_optimal_spectrogram_params(audio_sample_rate=16000, use_case='music'):
    """
    Get optimal spectrogram parameters for different use cases.
    
    Args:
        audio_sample_rate: Audio sample rate in Hz
        use_case: One of 'music', 'speech', 'general'
    
    Returns:
        dict: Optimal spectrogram parameters
    """
    
    # Base parameters optimized for 16kHz audio
    base_params = {
        'spec_n_fft': 2048,  # Provides good frequency resolution (~7.8 Hz per bin)
        'spec_win_length': 512,  # 32ms window at 16kHz
        'spec_power': 2,  # Power spectrogram
        'spec_center': True,
        'spec_pad_mode': 'reflect',
        'spec_aug_freq_mask': 27,  # Frequency masking parameter
        'spec_aug_time_mask': 100,  # Time masking parameter
        'spec_aug_stretch_factor': 0.8,  # Time stretching factor
    }
    
    # Adjust parameters based on use case
    if use_case == 'music':
        # Optimized for music generation
        music_params = {
            'spec_n_fft': 2048,  # High frequency resolution for music
            'spec_win_length': 512,  # Balance between time and frequency resolution
            'spec_hop_length': 128,  # 75% overlap for smooth spectrograms
            'spec_aug_freq_mask': 27,  # Moderate frequency masking for music
            'spec_aug_time_mask': 100,  # Moderate time masking
            'patch_size': 16,  # Standard patch size for music
        }
        base_params.update(music_params)
        
    elif use_case == 'speech':
        # Optimized for speech processing
        speech_params = {
            'spec_n_fft': 1024,  # Lower frequency resolution for speech
            'spec_win_length': 400,  # 25ms window (standard for speech)
            'spec_hop_length': 160,  # 10ms hop (standard for speech)
            'spec_aug_freq_mask': 15,  # Less frequency masking for speech
            'spec_aug_time_mask': 50,  # Less time masking for speech
            'patch_size': 8,  # Smaller patches for speech details
        }
        base_params.update(speech_params)
        
    elif use_case == 'general':
        # Balanced parameters for general audio
        general_params = {
            'spec_n_fft': 1536,  # Balanced frequency resolution
            'spec_win_length': 480,  # ~30ms window
            'spec_hop_length': 160,  # ~10ms hop
            'spec_aug_freq_mask': 20,  # Balanced frequency masking
            'spec_aug_time_mask': 75,  # Balanced time masking
            'patch_size': 12,  # Medium patch size
        }
        base_params.update(general_params)
    
    # Adjust parameters based on sample rate
    if audio_sample_rate != 16000:
        # Scale window and hop lengths proportionally
        scale_factor = audio_sample_rate / 16000
        base_params['spec_win_length'] = int(base_params['spec_win_length'] * scale_factor)
        if 'spec_hop_length' in base_params:
            base_params['spec_hop_length'] = int(base_params['spec_hop_length'] * scale_factor)
    
    # Calculate hop length if not specified
    if 'spec_hop_length' not in base_params:
        base_params['spec_hop_length'] = base_params['spec_win_length'] // 4
    
    return base_params


def create_optimized_audiospectrogram_transformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    use_case='music',
    audio_sample_rate=16000,
    support_variable_length=True,
    **kwargs
):
    """
    Create an AudioSpectrogramTransformer with optimized parameters for specific use cases.
    
    Args:
        dim: Model dimension
        depth: Transformer depth
        heads: Number of attention heads
        dim_head: Dimension per head
        use_case: One of 'music', 'speech', 'general'
        audio_sample_rate: Audio sample rate
        support_variable_length: Whether to support variable length audio
        **kwargs: Additional arguments
    
    Returns:
        AudioSpectrogramTransformer with optimized parameters
    """
    
    # Get optimal spectrogram parameters
    spec_params = get_optimal_spectrogram_params(audio_sample_rate, use_case)
    
    # Create the transformer with optimized parameters
    return AudioSpectrogramTransformer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        spec_n_fft=spec_params['spec_n_fft'],
        spec_win_length=spec_params['spec_win_length'],
        spec_hop_length=spec_params['spec_hop_length'],
        spec_aug_freq_mask=spec_params['spec_aug_freq_mask'],
        spec_aug_time_mask=spec_params['spec_aug_time_mask'],
        spec_aug_stretch_factor=spec_params['spec_aug_stretch_factor'],
        support_variable_length=support_variable_length,
        max_audio_length=audio_sample_rate * 20,  # 20 seconds max
        **kwargs
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#***************************************************************************************basic operation
#***************************************************************************************layernorm, starrelu, UnfoldReshape
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class UnfoldReshape(nn.Module):
    def __init__(self, dim, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        # self.weight = nn.Parameter(torch.randn(dim, kernel_size, kernel_size)*.02)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        B, C, H, W = x.shape
        # _, _, k = self.weight.shape
        # B,C*kk,H,W -> B,C,h*w,kk
        h, w = H // self.stride, W // self.stride
        x = self.unfold(x).view(B, C, -1, h * w).transpose(-2, -1)
        # x = x * self.weight.view(1, C, 1, k * k)
        return x


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
        self.dim=dim

    def forward(self, x):
        return x * self.scale

#***************************************************************************************basic module
#***************************************************************************************attention, cpe, eap

class Attention(nn.Module):
    """
    Vanilla self-attention. Relative position embedding is not used.
    """

    def __init__(self, dim, head_dim=16, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mode='spatial'):
        B, S, H, W, C = x.shape
        if mode == 'spatial':
            N = H * W
            qkv = self.qkv(x).reshape(B, S, N, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5) # (3 b s nh n hd)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v)
            x = rearrange(x, 'b s nh (h w) hd -> b s h w (nh hd)', h=H, w=W)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif mode == 'spectral':
            qkv = self.qkv(x).reshape(B, S, H, W, 3, self.num_heads, self.head_dim).permute(4, 0, 2, 3, 5, 1, 6) # (3 b h w nh s hd)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v)
            x = rearrange(x, 'b h w nh s hd -> b s h w (nh hd)')
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Cross_attention(nn.Module):
    """
    cross attention
    """

    def __init__(self, dim, head_dim=16, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, feat):
        B, S, H, W, C = query.shape

        # # spectral cross attention
        # q = rearrange(query, 'b (nh hd) s h w -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        # k = rearrange(feat, 'b (nh hd) s h w-> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        # v = rearrange(feat, 'b (nh hd) s h w -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x_spectral = (attn @ v)
        # x_spectral = rearrange(x_spectral, 'b h w nh s hd -> b (nh hd) s h w')
        #
        # # spatial cross attention
        # q = rearrange(query, 'b (nh hd) s h w -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        # k = rearrange(feat, 'b (nh hd) s h w -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        # v = rearrange(feat, 'b (nh hd) s h w -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x_spatial = (attn @ v)
        # x_spatial = rearrange(x_spatial, 'b s nh (h w) hd -> b (nh hd) s h w', h=H, w=W)

        # spectral cross attention
        q = rearrange(query, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        k = rearrange(feat, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        v = rearrange(feat, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_spectral = (attn @ v)
        x_spectral = rearrange(x_spectral, 'b h w nh s hd -> b s h w (nh hd)')

        # spatial cross attention
        q = rearrange(query, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        k = rearrange(feat, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        v = rearrange(feat, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_spatial = (attn @ v)
        x_spatial = rearrange(x_spatial, 'b s nh (h w) hd -> b s h w (nh hd)', h=H, w=W)

        # spectral + spatial
        x = x_spectral + x_spatial

        return x


class CPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cpe = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x):
        x = x + self.cpe(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)

        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLP_head(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=100, mlp_ratio=2, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


# efficient attention pooling
class EAP_v1(nn.Module):
    def __init__(self, dim, out_dim, head_dim=32, kernel_size=3, ratio=2):
        super().__init__()
        self.num_heads = out_dim // head_dim
        self.out_dim = out_dim
        self.scale = (head_dim) ** -0.5

        self.qkv = nn.Conv2d(dim, out_dim * 3, kernel_size=1, bias=False)
        self.down_q = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=ratio,
                      padding=kernel_size // 2, groups=out_dim, bias=False),
            LayerNorm(out_dim, data_format='channels_first')
        )
        self.unfold_k = nn.Sequential(
            UnfoldReshape(out_dim, kernel_size=kernel_size, stride=ratio,
                          padding=kernel_size // 2),
            LayerNorm(out_dim, data_format='channels_first')
        )
        self.unfold_v = nn.Sequential(
            UnfoldReshape(out_dim, kernel_size=kernel_size, stride=ratio,
                          padding=kernel_size // 2),
            LayerNorm(out_dim, data_format='channels_first')
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.out_dim, dim=1)

        q, k, v = self.down_q(q), self.unfold_k(k), self.unfold_v(v)

        B, C, h, w = q.shape
        c, n = C // self.num_heads, h * w
        q = q.view(B, self.num_heads, c, n)
        k = k.view(B, self.num_heads, c, n, -1)
        v = v.view(B, self.num_heads, c, n, -1)

        attn = torch.einsum('bhcn,bhcnk->bhnk', q, k)
        attn = (attn * self.scale).softmax(dim=-1)

        x = torch.einsum('bhnk,bhcnk->bhcn', attn, v)
        x = x.reshape(B, C, n).transpose(-2, -1)  # B, n, C
        x = self.proj(x)
        return x, h, w


class CFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = rearrange(x, 'b s h w c -> b c s h w')
        x = self.dwconv(x)
        x = rearrange(x, 'b c s h w -> b s h w c')

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x








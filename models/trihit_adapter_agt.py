from models.adapter_modules import *
from models.trihit_x import trihit_x

def centercrop(x, crop_size):
    b, c, l, h, w = x.shape
    bound = (h - crop_size) // 2
    hs, he = bound, bound + crop_size
    ws, we = bound, bound + crop_size
    crop_x = x[:,:,:, hs:he, ws:we]
    return crop_x

class MLP_3d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim, bias=False)  # pointwise/1x1 convs, implemented with linear layers
        self.act = StarReLU()
        self.pwconv2 = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        input = x # N L H W C
        x = x.permute(0, 4, 1, 2, 3)  # (N, L, H, W, C) -> (N, C, L, H, W)
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, L, H, W) -> (N, L, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x

class hitblock(nn.Module):
    """
    Triple structured hyperspectral image transformer block (trihit block),
    spectral attention and spatial attention are conducted in parallel way.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        # spectral attention
        self.norm1 = norm_layer(dim)
        self.spectral_attn = Attention(dim=dim)

        # spatial attention
        self.norm2 = norm_layer(dim)
        self.spatial_attn = Attention(dim=dim)

        # IRFFN
        self.norm3 = norm_layer(dim)
        self.mlp = MLP_3d(dim=dim)

    def forward(self, x):
        x_spe = x + self.spectral_attn(self.norm1(x), 'spectral')
        x_spa = x + self.spatial_attn(self.norm2(x), 'spatial')
        x = x_spe + x_spa
        x = x + self.mlp(self.norm3(x))
        return x


class hitlayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        # blocks
        self.blocks = nn.Sequential(*[
            hitblock(dim=dim)
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x

class Cross_attention_bridge(nn.Module):
    """
    cross attention bridge
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

        self.q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.attention_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, feat):
        query = self.q(query)
        B, S, H, W, C = query.shape
        feat = self.kv(feat)
        feat_k, feat_v = torch.split(feat, C, dim=-1)

        # spectral cross attention
        q = rearrange(query, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        k = rearrange(feat_k, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        v = rearrange(feat_v, 'b s h w (nh hd) -> b h w nh s hd', nh=self.num_heads, hd=self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_spectral = (attn @ v)
        x_spectral = rearrange(x_spectral, 'b h w nh s hd -> b s h w (nh hd)')

        # spatial cross attention
        q = rearrange(query, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        k = rearrange(feat_k, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        v = rearrange(feat_v, 'b s h w (nh hd) -> b s nh (h w) hd', nh=self.num_heads, hd=self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_spatial = (attn @ v)
        x_spatial = rearrange(x_spatial, 'b s nh (h w) hd -> b s h w (nh hd)', h=H, w=W)

        # spectral + spatial
        x = x_spectral + x_spatial
        x = self.norm(x)

        return x

class Out_attention_bridge(nn.Module):
    """
    cross attention bridge
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

        self.kv = nn.Linear(dim, self.attention_dim*2 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, out):
        B, S, H, W, C = query.shape

        # fusing out info
        out_b, out_s, out_c = out.shape
        out_k, out_v = self.kv(out).reshape(B, 2, self.num_heads, out_s, self.head_dim).unbind(1)
        update_q = rearrange(query, 'b s h w (nh hd) -> b nh (s h w) hd', nh=self.num_heads, hd=self.head_dim)
        attn = (update_q @ out_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ out_v)
        x= rearrange(x, 'b nh (s h w) hd -> b s h w (nh hd)', s = S, h=H, w=W)

        return x

"""
num_classes=num_classes,
bk_depths=[2, 2, 4, 2], bk_dims=[32, 64, 128, 256], bk_arch=['conv', 'hit', 'hit', 'hit'],
dp_rate=dp_rate,
ladder_depths=[1,1,2,1], ladder_dims=[32, 64, 128, 256], ladder_input_size=13
"""


class trihit_sdt(trihit_x):
    def __init__(self,
                 in_chans=1,
                 num_classes=100,
                 bk_depths=[2, 2, 4, 2],
                 bk_dims=[32, 64, 128, 256],
                 bk_arch=['conv', 'hit', 'hit', 'hit'],
                 dp_rate=0.15,
                 ladder_depths=[1, 2, 1],
                 ladder_dims=[64, 128, 256],
                 ladder_input_size=13,
                 head_dropout=0.0,
                 ):

        super().__init__(in_chans=in_chans, depths=bk_depths, dims=bk_dims, drop_path_rate=dp_rate, arch=bk_arch, add_classifer=False)

        self.adapter_downsample_layers = nn.ModuleList()
        self.adapter_stages = nn.ModuleList()
        self.cross_adapter_bridge = nn.ModuleList()
        self.out_adapter_bridge = nn.ModuleList()
        self.adapter_bridge = nn.ModuleList()
        self.out_fusions = nn.ModuleList()
        self.ladder_input_size = ladder_input_size

        stem = nn.Sequential(
            nn.Conv3d(in_chans, ladder_dims[0] // 2, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(ladder_dims[0] // 2),
            StarReLU(),
            nn.Conv3d(ladder_dims[0] // 2, ladder_dims[0], kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(ladder_dims[0]),
        )
        self.adapter_downsample_layers.append(stem)

        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(ladder_dims[i]),
                nn.Conv3d(ladder_dims[i], ladder_dims[i + 1], kernel_size=3, stride=2, padding=0),
            )
            self.adapter_downsample_layers.append(downsample_layer)

        for i in range(3):
            stage = hitlayer(dim=ladder_dims[i], depth=ladder_depths[i])
            self.adapter_stages.append(stage)

        for i in range(3):
            cross_bridge = Cross_attention_bridge(dim=ladder_dims[i])
            self.cross_adapter_bridge.append(cross_bridge)

        for i in range(3):
            out_bridge = Out_attention_bridge(dim=ladder_dims[i])
            self.out_adapter_bridge.append(out_bridge)

        for i in range(3):
            bridge = nn.Sequential(
                nn.Linear(ladder_dims[i], ladder_dims[i], bias=False),
                nn.LayerNorm(ladder_dims[i])
            )
            self.adapter_bridge.append(bridge)

        for i in range(3):
            fusion = nn.Sequential(
                nn.Linear(ladder_dims[-1], ladder_dims[i], bias=False),
                nn.LayerNorm(ladder_dims[i])
            )
            self.out_fusions.append(fusion)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.adapter_cls_norm = nn.LayerNorm(ladder_dims[-1])
        self.adapter_cls_head = MLP_head(ladder_dims[-1], num_classes, head_dropout=head_dropout)

    def forward(self, x):
        bk_x = x
        ladder_x = centercrop(x, self.ladder_input_size)
        bk_x = self.downsample_layers[0](bk_x)
        bk_x = self.stages[0](bk_x)

        bk_branch_res = []

        for i, [bk_down, bk_stage] in \
                enumerate(zip(self.downsample_layers[1:], self.stages[1:])):
            if i>0:
                bk_x = rearrange(bk_x, 'b s h w c-> b c s h w')

            bk_x = bk_down(bk_x)
            bk_x = rearrange(bk_x, 'b c s h w -> b s h w c')
            bk_x = bk_stage(bk_x)

            bk_branch_res.append(bk_x)

        bk_out = bk_x.mean([2,3])
        bk_out_mids = []
        for i in range(3):
            bk_out_i = self.out_fusions[i](bk_out)
            bk_out_mids.append(bk_out_i)

        for i, [adapter_down, adapter_stage, cross_bridge, out_bridge, bridge] in \
                enumerate(zip(self.adapter_downsample_layers, self.adapter_stages,
                              self.cross_adapter_bridge, self.out_adapter_bridge, self.adapter_bridge)):
            if i>0:
                ladder_x = rearrange(ladder_x, 'b s h w c-> b c s h w')

            ladder_x = adapter_down(ladder_x)
            ladder_x = rearrange(ladder_x, 'b c s h w -> b s h w c')

            bk_branch_x = bk_branch_res[i]
            ladder_x = ladder_x + (1 - self.alpha) *bridge(bk_branch_x) + self.alpha * cross_bridge(ladder_x, bk_branch_x) + out_bridge(ladder_x, bk_out_mids[i])
            ladder_x = adapter_stage(ladder_x)

        x = ladder_x.mean([1,2,3])
        x = self.adapter_cls_norm(x)
        out = self.adapter_cls_head(x)

        return out

"""
fusing features from different branches with bridge, which consists of a gate operation and a separable cross attention 
layer. In addition, features from the last stage of the pretrained branches are used in three stages of the new branches. 
"""

def trihit_cth_sdt_r5(num_classes=100, dp_rate=0.15):
    model = trihit_sdt(num_classes=num_classes,
                           bk_depths=[2, 2, 4, 2], bk_dims=[32, 64, 128, 256], bk_arch=['conv', 'hit', 'hit', 'hit'],
                           dp_rate=dp_rate,
                           ladder_depths=[1,2,1], ladder_dims=[64, 128, 256], ladder_input_size=13)
    return model


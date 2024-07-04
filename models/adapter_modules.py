from functools import partial
from models.basic_operators import *
from timm.models.layers import DropPath, trunc_normal_


class Extractor(nn.Module):
    def \
            __init__(self, dim, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = Cross_attention(dim=dim)

        self.ffn = CFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat):
        attn_x = self.attn(query, feat)
        query = query + attn_x
        query = query + self.drop_path(self.ffn(self.ffn_norm(query)))

        return query


class Injector(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = Cross_attention(dim=dim)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        attn_x = self.attn(query, feat)
        query = query + self.gamma * attn_x

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., cffn_ratio=0.25, init_values=0.):
        super().__init__()

        self.injector = Injector(dim=dim, norm_layer=norm_layer, init_values=init_values)
        self.extractor = Extractor(dim=dim, norm_layer=norm_layer, cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x, c, blocks):
        x = self.injector(query=x, feat=c)
        x = blocks(x)
        c = self.extractor(query=c, feat=x)

        return x, c

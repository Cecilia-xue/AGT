from models.basic_operators import *
from models.adapter_modules import *
from models.trihit_x import trihit_x

# the traditional adapeter architecture adopted in LWNet. Only the classification head is re-intialized and fine-tuned
class trihit_ft_adapter(trihit_x):
    def __init__(self,
                 in_chans=1,
                 num_classes=100,
                 depths=[2, 2, 4, 2],
                 dims=[64, 128, 320, 512],
                 arch=['conv', 'hit', 'hit', 'hit'],
                 head_dropout=0.0,
                 ):

        super().__init__(in_chans=in_chans, depths=depths, dims=dims, drop_path_rate=0.15, arch=arch, add_classifer=False)

        self.adapter_cls_norm = nn.LayerNorm(dims[-1])
        self.adapter_cls_head = MLP_head(dims[-1], num_classes, head_dropout=head_dropout)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.adapter_cls_norm(x)
        out = self.adapter_cls_head(x)

        return out

def trihit_cth_ft(num_classes=100, dp_rate=0.0):
    model = trihit_ft_adapter(num_classes=num_classes, depths=[2, 2, 4, 2], dims=[32, 64, 128, 256], arch=['conv', 'hit', 'hit', 'hit'],)
    return model



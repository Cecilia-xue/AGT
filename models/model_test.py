import torch
from models.trihit_adapter_agt import trihit_cth_sdt_r5

input = torch.randn(4, 1, 200, 27, 27).cuda()
model = trihit_cth_sdt_r5().cuda()
output = model(input)
print('done')


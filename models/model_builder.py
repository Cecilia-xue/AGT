from models.trihit_x import trihit_cth
from models.trihit_adapter_ft import trihit_cth_ft
from models.trihit_adapter_clip import trihit_cth_clip
from models.trihit_adapter_lora import trihit_cth_lora
from models.trihit_adapter_lst import trihit_cth_lst
from models.trihit_adapter_agt import trihit_cth_sdt_r5
models = {
    'trihit_cth': trihit_cth,
    'trihit_cth_ft': trihit_cth_ft,
    'trihit_cth_clip': trihit_cth_clip,
    'trihit_cth_lora': trihit_cth_lora,
    'trihit_cth_lst': trihit_cth_lst,
    'trihit_cth_sdt_r5': trihit_cth_sdt_r5,
}

def model_builder(model_name, num_classes, args):
    model = models[model_name](num_classes=num_classes, dp_rate=args.dp_rate)
    return model

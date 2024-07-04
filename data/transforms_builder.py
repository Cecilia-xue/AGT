import torch
import random
from torchvision import transforms

class flip:
    def __init__(self, p=0.5):
        self.p=p

    def __call__(self, x):
        # s h w
        use_flip = random.random() < self.p
        if use_flip:
            p = random.random()
            if p < 0.33:
                x = torch.flip(x, dims=[2])
            elif p < 0.66:
                x = torch.flip(x, dims=[1])
            else:
                x = torch.flip(x, dims=[1, 2])
        return x


class rotate:
    def __init__(self, p=0.5):
        self.p=p

    def __call__(self, x):
        # s h w
        use_rotate = random.random() < self.p
        if use_rotate:
            rotation_repeat = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, rotation_repeat, (1, 2))

        return x


class add_noise:
    def __init__(self, p=0.5, std=0.01):
        self.p=p
        self.std=std

    def __call__(self, x):
        # b c s h w
        if random.random() < self.p:
            x = x + torch.randn_like(x)*self.std
        return x


class cutout:
    def __init__(self, p=0.2, pixel_ratio=0.3, spectral_ratio=0.1):
        self.p=p
        self.pixel_r=pixel_ratio
        self.spectral_r=spectral_ratio

    def __call__(self, x):
        use_cutout = random.random() < self.p
        if use_cutout:
            s, h, w = x.size()
            spe_l = int(s*self.spectral_r)

            for i in range(h):
                for j in range(w):
                    if random.random() < self.pixel_r:
                        s_s = random.randint(0, s-1-spe_l)
                        s_e = s_s + spe_l
                        x[s_s:s_e, i, j] = 0
        return x


class rcutout:
    def __init__(self, p=0.2, pixel_ratio=0.3, spectral_ratio=0.1):
        self.p=p
        self.pixel_r=pixel_ratio
        self.spectral_r=spectral_ratio

    def __call__(self, x):
        use_cutout = random.random() < self.p
        if use_cutout:
            s, h, w = x.size()
            ins_pixel_r = random.uniform(0.01, self.pixel_r)
            ins_spectral_r = random.uniform(0.01, self.spectral_r)
            spe_l = int(s*ins_spectral_r)

            for i in range(h):
                for j in range(w):
                    if random.random() < ins_pixel_r:
                        s_s = random.randint(0, s-1-spe_l)
                        s_e = s_s + spe_l
                        x[s_s:s_e, i, j] = 0
        return x


def transforms_builder(args):
    t=[]

    if args.flip is not None:
        t.append(flip(args.flip))
    if args.rotate is not None:
        t.append(rotate(args.rotate))
    if args.add_noise is not None:
        t.append(add_noise(args.add_noise[0], args.add_noise[1]))
    if args.cutout is not None:
        t.append(cutout(args.cutout[0], args.cutout[1], args.cutout[2]))
    if args.rcutout is not None:
        t.append(rcutout(args.rcutout[0], args.rcutout[1], args.rcutout[2]))

    return transforms.Compose(t)

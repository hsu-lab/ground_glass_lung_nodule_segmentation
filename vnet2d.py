"""2D V-Net (Milletari et al., 2016) and a 2D adaptation of Dig-CS-VNet
(Sci Rep 2024, doi:10.1038/s41598-024-55178-3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ("vnet2d", "digcs_vnet2d")


def _norm(c, kind):
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "instance":
        return nn.InstanceNorm2d(c, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, c), c)
    raise ValueError(kind)


def _act(c, kind):
    if kind == "prelu":
        return nn.PReLU(c)
    if kind == "elu":
        return nn.ELU(alpha=1.0)
    raise ValueError(kind)

class DigSep(nn.Module):

    def __init__(self, n_bits=4):
        super().__init__()
        self.n_bits = n_bits

    @torch.no_grad()
    def forward(self, x):
        v = torch.clamp(torch.floor(x.float() * 255.0), 0, 255)
        planes = []
        for b in range(7, 7 - self.n_bits, -1):
            t = float(2 ** b)
            p = (v >= t).float()
            planes.append(p)
            v = v - p * t
        return torch.cat(planes, dim=1).to(x.dtype)


class ChannelAttention(nn.Module):
    """CBAM channel attention: shared MLP over global avg- and max-pooled features."""

    def __init__(self, c, reduction=8):
        super().__init__()
        hidden = max(c // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, c, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)


class SpatialAttention(nn.Module):
    """CBAM spatial attention: channel-wise avg & max -> concat -> 3x3 conv -> sigmoid
    (the paper uses a 3x3x3 conv in 3D; 3x3 here)."""

    def __init__(self, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    def __init__(self, c, reduction=8, kernel=3):
        super().__init__()
        self.ca = ChannelAttention(c, reduction)
        self.sa = SpatialAttention(kernel)

    def forward(self, x):
        return self.sa(self.ca(x))

# V-Net blocks
class ConvStack(nn.Module):
    def __init__(self, c, n, norm, act):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(c, c, 5, padding=2, bias=False), _norm(c, norm)]
            if i < n - 1:
                layers.append(_act(c, act))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class InputTransition(nn.Module):

    def __init__(self, in_ch, out_ch, norm, act, dig_sep=False):
        super().__init__()
        self.out_ch, self.dig_sep = out_ch, dig_sep
        self.sep = DigSep(4) if dig_sep else None
        conv_in = in_ch + 4 if dig_sep else in_ch
        self.conv = nn.Sequential(nn.Conv2d(conv_in, out_ch, 5, padding=2, bias=False), _norm(out_ch, norm))
        self.act = _act(out_ch, act)

    def forward(self, x):
        if self.dig_sep:
            planes = self.sep(x)                                     # (B, 4, H, W)
            y = self.conv(torch.cat([x, planes], dim=1))
            half = self.out_ch // 2
            res = torch.cat([planes.repeat(1, half // planes.shape[1], 1, 1),
                             x.repeat(1, half // x.shape[1], 1, 1)], dim=1)
        else:
            y = self.conv(x)
            res = x.repeat(1, self.out_ch // x.shape[1], 1, 1)
        return self.act(y + res)


class DownTransition(nn.Module):
    def __init__(self, in_ch, n_convs, norm, act, dropout=0.0, cbam=False):
        super().__init__()
        out_ch = in_ch * 2
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False),
                                  _norm(out_ch, norm), _act(out_ch, act))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ops = ConvStack(out_ch, n_convs, norm, act)
        self.act = _act(out_ch, act)
        self.att = CBAM(out_ch) if cbam else nn.Identity()  # "at the end of each down-sampling layer"

    def forward(self, x):
        d = self.down(x)
        return self.att(self.act(self.ops(self.drop(d)) + d))


class UpTransition(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs, norm, act, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch // 2, 2, stride=2, bias=False),
                                _norm(out_ch // 2, norm), _act(out_ch // 2, act))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ops = ConvStack(out_ch, n_convs, norm, act)
        self.act = _act(out_ch, act)

    def forward(self, x, skip):
        cat = torch.cat([self.up(self.drop(x)), skip], dim=1)
        return self.act(self.ops(cat) + cat)


class OutputTransition(nn.Module):
    def __init__(self, in_ch, n_classes, norm, act):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, 5, padding=2, bias=False),
                                  _norm(in_ch, norm), _act(in_ch, act),
                                  nn.Conv2d(in_ch, n_classes, 1))

    def forward(self, x):
        return self.conv(x)


class VNet2D(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, norm="batch", dropout=0.2,
                 act="prelu", dig_sep=False, cbam=False):
        super().__init__()
        self.in_tr = InputTransition(in_channels, 16, norm, act, dig_sep)
        self.down32 = DownTransition(16, 1, norm, act, cbam=cbam)
        self.down64 = DownTransition(32, 2, norm, act, cbam=cbam)
        self.down128 = DownTransition(64, 3, norm, act, dropout, cbam=cbam)
        self.down256 = DownTransition(128, 3, norm, act, dropout, cbam=cbam)
        self.up256 = UpTransition(256, 256, 3, norm, act, dropout)
        self.up128 = UpTransition(256, 128, 3, norm, act, dropout)
        self.up64 = UpTransition(128, 64, 2, norm, act)
        self.up32 = UpTransition(64, 32, 1, norm, act)
        self.out_tr = OutputTransition(32, n_classes, norm, act)

    def forward(self, x):
        o16 = self.in_tr(x)
        o32 = self.down32(o16)
        o64 = self.down64(o32)
        o128 = self.down128(o64)
        o256 = self.down256(o128)
        x = self.up256(o256, o128)
        x = self.up128(x, o64)
        x = self.up64(x, o32)
        x = self.up32(x, o16)
        return self.out_tr(x)


def build_model(name="vnet2d", norm="batch", dropout=0.2):
    if name == "vnet2d":
        return VNet2D(1, 1, norm=norm, dropout=dropout, act="prelu", dig_sep=False, cbam=False)
    if name == "digcs_vnet2d":
        return VNet2D(1, 1, norm=norm, dropout=dropout, act="elu", dig_sep=True, cbam=True)
    raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")


if __name__ == "__main__":
    for n in MODEL_NAMES:
        m = build_model(n)
        y = m(torch.rand(2, 1, 64, 64))
        print(f"{n:14s} out {tuple(y.shape)}  params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")

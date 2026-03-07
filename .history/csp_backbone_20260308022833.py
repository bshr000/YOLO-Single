"""
CSP (Cross Stage Partial) Backbone Implementation
"""
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        d = depth_multiple
        w = width_multiple
        
        # Channels configuration based on YOLOv8n/s/m/l/x
        # Base channels: [64, 128, 256, 512, 1024]
        c = [64, 128, 256, 512, 1024]
        c = [int(x * w) for x in c]
        
        # Layers configuration
        # n = [3, 6, 6, 3] (number of repeats)
        n = [3, 6, 6, 3]
        n = [max(round(x * d), 1) if x > 1 else x for x in n]

        # 0: Stem P1/2  (stride 2)
        self.stem = nn.Sequential(
            Conv(3, c[0], 3, 2), # P1/2
        )
        
        # 1: P2/4 (stride 2)
        self.layer1 = nn.Sequential(
            Conv(c[0], c[1], 3, 2),
            C2f(c[1], c[1], n=n[0], shortcut=True)
        )
        
        # 2: P3/8 (stride 2) - Output C3
        self.layer2 = nn.Sequential(
            Conv(c[1], c[2], 3, 2),
            C2f(c[2], c[2], n=n[1], shortcut=True)
        )
        
        # 3: P4/16 (stride 2) - Output C4
        self.layer3 = nn.Sequential(
            Conv(c[2], c[3], 3, 2),
            C2f(c[3], c[3], n=n[2], shortcut=True)
        )
        
        # 4: P5/32 (stride 2) - Output C5
        self.layer4 = nn.Sequential(
            Conv(c[3], c[4], 3, 2),
            C2f(c[4], c[4], n=n[3], shortcut=True),
            SPPF(c[4], c[4], 5)
        )
        
        self.out_channels = [c[2], c[3], c[4]]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return [c3, c4, c5]

if __name__ == "__main__":
    # Test
    model = CSPDarknet(depth_multiple=0.33, width_multiple=0.5) # YOLOv8s config
    x = torch.randn(1, 3, 640, 640)
    features = model(x)
    for i, f in enumerate(features):
        print(f"P{i+3} shape: {f.shape}")


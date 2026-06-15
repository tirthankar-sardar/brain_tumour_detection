import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ── Attention Gate ────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Additive attention gate.
    g  : gating signal  (B, g_ch, H, W)   — from the decoder (deeper, smaller)
    x  : skip feature   (B, x_ch, H', W') — from the encoder (shallower, larger)
    F_int: intermediate channel dim for the attention computation.
    """
    def __init__(self, g_ch, x_ch, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, F_int, kernel_size=1, bias=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, F_int, kernel_size=1, stride=1, bias=True),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample g to match x spatial size if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)          # (B, 1, H, W) attention map
        return x * psi               # attended skip


# ── Double Conv block ─────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# ── Decoder block with Attention Gate ────────────────────────────────────────
class AttDecoderBlock(nn.Module):
    """
    1. Upsample (ConvTranspose2d) the deeper feature map.
    2. Compute attention-gated skip via AttentionGate.
    3. Concatenate upsampled + attended-skip → DoubleConv.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.att = AttentionGate(g_ch=in_ch // 2, x_ch=skip_ch, F_int=skip_ch // 2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)                        # upsample
        skip = self.att(g=x, x=skip)             # attention-weighted skip
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── Attention U-Net ───────────────────────────────────────────────────────────
class AttentionUNet(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder and attention gates on skip connections.

    Input : (B, 3, H, W)  — RGB
    Output (eval) : (B, 1, H, W)  — raw logits
    Output (train): [(B,1,H,W), (B,1,H/2,W/2), (B,1,H/4,W/4), (B,1,H/8,W/8)]
                     main + 3 deep-supervision aux heads
    """
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.n_classes = n_classes

        # ── ResNet-34 encoder (torchvision) ──────────────────────────────────
        backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)

        # Stem: conv1 + bn + relu  →  64ch, H/2
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        # Pool                     →  64ch, H/4
        self.pool  = backbone.maxpool
        # ResNet stages
        self.enc1  = backbone.layer1   # 64ch,  H/4
        self.enc2  = backbone.layer2   # 128ch, H/8
        self.enc3  = backbone.layer3   # 256ch, H/16
        self.enc4  = backbone.layer4   # 512ch, H/32

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(512, 1024)

        # ── Decoder with Attention Gates ──────────────────────────────────────
        # AttDecoderBlock(in_ch_from_prev, skip_ch, out_ch)
        self.dec4 = AttDecoderBlock(1024, 256, 512)   # H/16
        self.dec3 = AttDecoderBlock(512,  128, 256)   # H/8
        self.dec2 = AttDecoderBlock(256,   64,  128)  # H/4
        self.dec1 = AttDecoderBlock(128,   64,   64)  # H/2  (skip = enc0 stem)

        # Final upsample H/2 → H
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(32, 32),
        )

        # ── Segmentation head ─────────────────────────────────────────────────
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        # ── Deep supervision auxiliary heads ──────────────────────────────────
        self.ds4 = nn.Conv2d(512, n_classes, 1)  # after dec4 → H/16
        self.ds3 = nn.Conv2d(256, n_classes, 1)  # after dec3 → H/8
        self.ds2 = nn.Conv2d(128, n_classes, 1)  # after dec2 → H/4

    def forward(self, x):
        H, W = x.shape[2:]

        # ── Encoder ───────────────────────────────────────────────────────────
        s0  = self.enc0(x)          # 64ch,  H/2
        p   = self.pool(s0)         # 64ch,  H/4
        s1  = self.enc1(p)          # 64ch,  H/4
        s2  = self.enc2(s1)         # 128ch, H/8
        s3  = self.enc3(s2)         # 256ch, H/16
        s4  = self.enc4(s3)         # 512ch, H/32

        # ── Bottleneck ────────────────────────────────────────────────────────
        b   = self.bottleneck(s4)   # 1024ch, H/32

        # ── Decoder ───────────────────────────────────────────────────────────
        d4  = self.dec4(b,  s3)     # 512ch,  H/16
        d3  = self.dec3(d4, s2)     # 256ch,  H/8
        d2  = self.dec2(d3, s1)     # 128ch,  H/4
        d1  = self.dec1(d2, s0)     # 64ch,   H/2
        d0  = self.dec0(d1)         # 32ch,   H

        main_out = self.final(d0)   # 1ch,    H

        if self.training:
            aux4 = F.interpolate(self.ds4(d4), size=(H, W), mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.ds3(d3), size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.ds2(d2), size=(H, W), mode='bilinear', align_corners=False)
            return [main_out, aux2, aux3, aux4]   # main + 3 aux (shallow → deep)
        return main_out
    
import torch

def load_model(weights_path):
    model = AttentionUNet(
        n_classes=1,
        pretrained=False
    )

    state = torch.load(
        weights_path,
        map_location="cpu"
    )

    model.load_state_dict(state)
    model.eval()

    return model
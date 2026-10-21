"""
model_zoo.py -- drop-in alternative architectures for the robustness experiment.

All return a 2D network: in_channels=1, out_channels=C.NUM_CLASSES (4), with dropout
kept >0 so an MC-dropout baseline stays available. Same input/output contract as the
original build_unet(), so train/extract code is architecture-agnostic.

    from model_zoo import build_model
    net = build_model("unet")        # baseline (matches model.py)
    net = build_model("segresnet")   # residual + autoencoder-reg family
    net = build_model("attnunet")    # attention U-Net
    net = build_model("swinunetr")   # transformer (different inductive bias)
"""
from monai.networks.nets import UNet, SegResNet, AttentionUnet, SwinUNETR
import config as C

ARCHS = ["unet", "segresnet", "attnunet", "swinunetr"]


def build_model(arch: str):
    arch = arch.lower()
    if arch == "unet":
        return UNet(
            spatial_dims=2, in_channels=1, out_channels=C.NUM_CLASSES,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            num_res_units=2, dropout=0.1,
        )
    if arch == "segresnet":
        return SegResNet(
            spatial_dims=2, in_channels=1, out_channels=C.NUM_CLASSES,
            init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
            dropout_prob=0.1,
        )
    if arch == "attnunet":
        return AttentionUnet(
            spatial_dims=2, in_channels=1, out_channels=C.NUM_CLASSES,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            dropout=0.1,
        )
    if arch == "swinunetr":
        # Transformer backbone. Constraints, both satisfied by the pipeline:
        # input spatial dims divisible by 32 (256/32=8) -- now checked inside
        # forward(), so img_size is NO LONGER a constructor argument -- and
        # feature_size divisible by 12 (24). drop_rate>0 keeps the MC-dropout
        # baseline available, as above.
        # NOTE: MONAI 1.5+ removed the img_size argument entirely; do not pass it.
        return SwinUNETR(
            in_channels=1, out_channels=C.NUM_CLASSES,
            feature_size=24, spatial_dims=2, drop_rate=0.1, use_checkpoint=False,
        )
    raise ValueError(f"unknown arch '{arch}'. choose from {ARCHS}")


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 1, *C.PATCH_SIZE)
    for a in ARCHS:
        net = build_model(a)
        y = net(x)
        n = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"{a:10s} out={tuple(y.shape)}  params={n:.2f}M")

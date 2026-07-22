import torch, sys

path = sys.argv[1] if len(sys.argv) > 1 else "pgml_best.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"file        : {path}")
print(f"epoch       : {ckpt['epoch']}")
print(f"val_loss    : {ckpt['val_loss']:.6e}")
print(f"lambda      : {ckpt['lambda']:.6e}")
print(f"n_params    : {sum(v.numel() for v in ckpt['model_state'].values()):,}")
print(f"args        : {ckpt['args']}")
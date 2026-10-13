"""fleet_v6.apply_norm must match train_knee's ImageNet normalisation, and be a no-op for old checkpoints."""
import torch

SRC = open("kaggle_base954_fleet/fleet_v6.py").read()
start = SRC.index("def apply_norm")
ns = {"torch": torch}
exec(SRC[start:SRC.index("\n\n", start)], ns)
apply_norm = ns["apply_norm"]


def test_none_is_identity():
    x = torch.rand(2, 3, 8, 8)
    assert torch.equal(apply_norm(x, "none"), x)


def test_imagenet_matches_train_knee():
    x = torch.full((1, 3, 4, 4), 0.5)
    exp = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(apply_norm(x, "imagenet")[0, :, 0, 0], exp, atol=1e-6)

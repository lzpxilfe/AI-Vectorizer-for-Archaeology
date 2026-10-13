"""Masked dual-head BCE/Dice and centreline topology loss."""


def _soft_erode(value):
    import torch
    import torch.nn.functional as functional

    vertical = -functional.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
    horizontal = -functional.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_open(value):
    import torch.nn.functional as functional

    return functional.max_pool2d(_soft_erode(value), 3, stride=1, padding=1)


def _soft_skeleton(value, iterations=3):
    import torch

    skeleton = torch.relu(value - _soft_open(value))
    for _ in range(iterations):
        value = _soft_erode(value)
        delta = torch.relu(value - _soft_open(value))
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability, target, valid_mask, *, iterations=3):
    """Exclude ignore-adjacent topology rather than inventing cut endpoints."""
    import torch
    import torch.nn.functional as functional

    valid = valid_mask.to(probability.dtype)
    margin = iterations + 2
    invalid = functional.pad(1.0 - valid, (margin,) * 4, value=1.0)
    safe = 1.0 - functional.max_pool2d(invalid, 2 * margin + 1, stride=1)
    if not bool(safe.any()):
        return probability.sum() * 0.0
    predicted_skeleton = _soft_skeleton(probability * valid, iterations) * safe
    target_skeleton = _soft_skeleton(target * valid, iterations) * safe
    epsilon = 1e-6
    precision = ((predicted_skeleton * target).sum() + epsilon) / (predicted_skeleton.sum() + epsilon)
    sensitivity = ((target_skeleton * probability).sum() + epsilon) / (target_skeleton.sum() + epsilon)
    return 1.0 - (2.0 * precision * sensitivity + epsilon) / (precision + sensitivity + epsilon)


def contour_loss(logits, centerline, other_ink, valid_mask, *, topology_weight=0.2):
    """Return loss and detached component metrics; ignore logits have no effect."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits must have two full-resolution heads")
    expected = (logits.shape[0], logits.shape[2], logits.shape[3])
    if any(tuple(value.shape) != expected for value in (centerline, other_ink, valid_mask)):
        raise ValueError("targets and valid_mask must match logit batch/grid")
    targets = torch.stack((centerline, other_ink), dim=1).to(logits.dtype)
    valid = valid_mask.unsqueeze(1).to(logits.dtype)
    if not bool(torch.isfinite(targets).all()) or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("targets must be finite scores in [0,1]")
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, {"bce": 0.0, "dice": 0.0, "cldice": 0.0}
    bce = (functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid).sum() / (valid.sum() * 2.0)
    probabilities = torch.sigmoid(logits)
    dimensions = (0, 2, 3)
    intersection = (probabilities * targets * valid).sum(dim=dimensions)
    denominator = ((probabilities + targets) * valid).sum(dim=dimensions)
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    topology = soft_cldice_loss(probabilities[:, :1], targets[:, :1], valid)
    loss = bce + dice + float(topology_weight) * topology
    return loss, {"bce": float(bce.detach()), "dice": float(dice.detach()), "cldice": float(topology.detach())}


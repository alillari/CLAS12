"""Explicit, opt-in partial backbone tuning for momentum adapters."""
import math
import torch


def validate_backbone_finetuning(params, pretrain):
    count = params.get("backbone_unfreeze_last_n", 0)
    scale = params.get("backbone_lr_scale", 0.1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("backbone_unfreeze_last_n must be a nonnegative integer")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("backbone_lr_scale must be finite and positive")
    if count:
        if not pretrain:
            raise ValueError("Backbone unfreezing requires pretrained mode")
        if params.get("mambaversion") not in {"mamba1", "mamba2"}:
            raise ValueError("Partial backbone tuning supports Mamba backbones only")
        if count > params["num_layers_backbone"]:
            raise ValueError("Cannot unfreeze more blocks than the backbone contains")
        if int(params.get("world_size", 1)) != 1:
            raise ValueError("Partial backbone tuning currently requires one process")
    return count


def configure_backbone(model, count):
    """Freeze everything except the final count residual blocks.

    Keep dropout off in both frozen and tuned arms; eval mode does not prevent
    parameter gradients. Embedding, output norm and pretraining head stay frozen.
    """
    model.requires_grad_(False)
    model.eval()
    if count:
        if not hasattr(model, "mamba_layers") or count > len(model.mamba_layers):
            raise ValueError("Requested residual blocks do not exist")
        for block in model.mamba_layers[-count:]:
            block.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def backbone_features(model, model_input, tune=False):
    # A frozen prefix has no grad-requiring inputs or parameters, so autograd
    # starts at the first tuned block without retaining prefix activations.
    with torch.set_grad_enabled(torch.is_grad_enabled() and tune):
        _, features, _ = model(model_input, return_z=True)
    return torch.stack(features)


def restore_tuned_backbone(model, checkpoint, expected_count):
    count = checkpoint.get("backbone_unfreeze_last_n", 0)
    state = checkpoint.get("backbone_state_dict")
    if count != expected_count or bool(count) != (state is not None):
        raise ValueError("Checkpoint backbone tuning contract does not match configuration/weights")
    if count:
        model.load_state_dict(state, strict=True)


class ScaledGroupScheduler:
    """Preserve per-group LR ratios for the existing scalar-LR schedulers."""
    def __init__(self, scheduler, scales):
        self.scheduler = scheduler
        self.scales = list(scales)
        if len(scales) != len(scheduler.optimizer.param_groups):
            raise ValueError("One LR scale is required per optimizer group")
        self._apply()

    def _apply(self):
        for group, scale in zip(self.scheduler.optimizer.param_groups, self.scales):
            group["lr"] *= scale

    def step(self):
        self.scheduler.step()
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.scheduler.optimizer.param_groups]

    def state_dict(self):
        return {"scales": self.scales, "scheduler": self.scheduler.state_dict(),
                "unscaled_lrs": [lr / scale for lr, scale in
                    zip(self.get_last_lr(), self.scales)]}

    def load_state_dict(self, state):
        if state["scales"] != self.scales:
            raise ValueError("Checkpoint learning-rate scales differ")
        self.scheduler.load_state_dict(state["scheduler"])
        for group, lr, scale in zip(self.scheduler.optimizer.param_groups,
                                     state["unscaled_lrs"], self.scales):
            group["lr"] = lr * scale

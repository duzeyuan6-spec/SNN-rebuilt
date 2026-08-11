import torch
import torch.nn as nn


E_MAC_PJ = 4.6
E_AC_PJ = 0.9


def conv2d_macs(module, out):
    b, cout, h, w = out.shape
    kh, kw = module.kernel_size
    cin = module.in_channels
    groups = module.groups
    return b * cout * h * w * (cin // groups) * kh * kw


def linear_macs(module, out):
    batch = out.numel() // module.out_features
    return batch * module.in_features * module.out_features


@torch.no_grad()
def estimate_ops(model, sample_y):
    macs = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if isinstance(output, torch.Tensor):
                macs += conv2d_macs(module, output)
        elif isinstance(module, nn.Linear):
            if isinstance(output, torch.Tensor):
                macs += linear_macs(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    _ = model(sample_y)

    for h in hooks:
        h.remove()

    firing_rate = model.firing_rate() if hasattr(model, "firing_rate") else 0.0
    if getattr(model, "kind", "ann") == "spike":
        # Standard SNN estimate: spike-triggered convolution operations are ACs.
        # The final membrane readout is already much smaller than hidden feature
        # convolutions, so we report the sparse event-driven estimate.
        acs = macs * firing_rate
        energy_pj = acs * E_AC_PJ
        return {"macs": macs, "acs": acs, "firing_rate": firing_rate, "energy_pj": energy_pj}

    energy_pj = macs * E_MAC_PJ
    return {"macs": macs, "acs": 0, "firing_rate": 0.0, "energy_pj": energy_pj}

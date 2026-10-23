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
    """Arithmetic-only estimate, not hardware energy; excludes memory/neuron work."""
    from .cs import BlockCSOperator
    rows, hooks, observers = [], [], []
    modules = dict(model.named_modules())

    def hook(name, module, inputs, output):
        x = inputs[0]
        if isinstance(module, nn.ConvTranspose2d) or getattr(module,'transposed_burst_readout',False):
            ops = x.numel() * (module.out_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Conv2d):
            ops = conv2d_macs(module, output)
        else:
            ops = linear_macs(module, output)
        parent = modules.get(name.split('.hidden.')[0])
        candidate = getattr(module, 'event_driven_candidate', False) or (
            '.hidden.' in name and hasattr(parent, 'lif_hidden'))
        binary = candidate and bool(torch.all((x == 0) | (x == 1)))
        signed = getattr(module,'signed_event_candidate',False) and bool(torch.all((x==-1)|(x==0)|(x==1)))
        burst=False;event_multiplicity=None
        if getattr(module,'burst_event_candidate',False):
            node=module.burst_node
            counts=x/node.quantum
            burst=bool(torch.allclose(counts,counts.round(),atol=1e-4,rtol=1e-5)) and bool((counts.abs()<=node.levels+1e-4).all())
            if burst:event_multiplicity=float(counts.abs().mean())
        event_input = binary or signed or burst
        fr = (event_multiplicity/node.levels if burst else float(x.float().abs().mean())) if event_input else None
        rows.append(dict(layer=name, dense_equivalent_macs=ops,
                         analog_macs=0 if event_input else ops,
                         hypothetical_acs=ops * (event_multiplicity if burst else fr) if event_input else 0,
                         event_encoding='signed_burst_counts' if burst else ('signed_ternary' if signed else ('binary' if binary else None)),
                         events_per_input=event_multiplicity,
                         input_firing_rate=fr))

    def sensed(ops):
        rows.append(dict(layer='measurement_operator', dense_equivalent_macs=ops,
                         analog_macs=ops, hypothetical_acs=0, input_firing_rate=None))

    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                hooks.append(module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook(n, mod, inp, out)))
            if isinstance(module, BlockCSOperator):
                observers.append((module, getattr(module, '_operation_observer', None)))
                module._operation_observer = sensed
        model(sample_y)
    finally:
        for h in hooks:
            h.remove()
        for module, previous in observers:
            module._operation_observer = previous
    macs = sum(r['dense_equivalent_macs'] for r in rows)
    analog = sum(r['analog_macs'] for r in rows)
    acs = sum(r['hypothetical_acs'] for r in rows)
    return dict(macs=macs, analog_macs=analog, acs=acs,
                dense_backend_macs=macs,
                execution_note='Standard torch convolutions execute dense kernels even for binary inputs; AC cost assumes a hypothetical event-driven implementation',
                firing_rate=model.firing_rate() if hasattr(model, 'firing_rate') else 0.,
                energy_pj=analog * E_MAC_PJ + acs * E_AC_PJ,
                dense_arithmetic_energy_pj=macs * E_MAC_PJ,
                scope='conv_linear_measurement_arithmetic_only', hardware_measured=False,
                excluded=['memory traffic', 'neuron updates', 'normalization', 'bias', 'activation'],
                layers=rows)

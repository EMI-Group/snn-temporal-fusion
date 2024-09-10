import torch
import torch.nn as nn
import temporal_fusion_kernel


class FusedLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tx, time_step, decay, threshold, rest, use_tv: bool=False):
        ctx.time_step = time_step
        ctx.decay = decay
        ctx.threshold = threshold
        ty = torch.zeros_like(tx)
        if use_tv:
            v_tv = torch.zeros_like(tx)
        else:
            v_tv = torch.zeros_like(tx[0])
        temporal_fusion_kernel.fusedForwardLIF(tx, v_tv, ty, time_step, decay, threshold, rest, use_tv)
        ctx.tv = v_tv
        ctx.save_for_backward(ty)
        return ty

    @staticmethod
    def backward(ctx, grad_ty):
        ty, = ctx.saved_tensors
        time_step = ctx.time_step
        decay = ctx.decay
        threshold = ctx.threshold
        tv = ctx.tv
        grad_tx = torch.zeros_like(grad_ty)
        temporal_fusion_kernel.fusedBackwardLIF(grad_ty, grad_tx, ty, tv, time_step, decay, threshold)
        return grad_tx, None, None, None, None, None


class LIF(nn.Module):
    def __init__(self, decay: float=0.2, threshold: float=0.3, rest: float=0.0, time_step: int=None): 
        super(LIF, self).__init__()
        self.decay = decay
        self.threshold = threshold
        self.rest = rest
        self.time_step = time_step

    def forward(self, tx):
        ty = FusedLIF.apply(tx, self.time_step, self.decay, self.threshold, self.rest, self.training)
        return ty


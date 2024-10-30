import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from mmcv.ops import DeformConv2d
import torch.nn as nn
class CustomDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        super(CustomDeformConv2d, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups)

    def forward(self, input, offset):
        return self.conv(input, offset)



class DeformConvFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        input, 
        offset, 
        weight,
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        deformable_groups=1, 
        im2col_step=64
    ):
        ctx.save_for_backward(input, weight, offset)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups

        # 这里可以放置自定义实现的可变形卷积操作
        # 在纯 Python 中可能需要手动计算偏移后的采样位置
        output = DeformConv2d(in_channels=input.size(1),
                              out_channels=weight.size(0),
                              kernel_size=weight.size(2),
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              deformable_groups=deformable_groups)(input, offset)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, offset = ctx.saved_tensors

        # 计算输入的梯度
        grad_input = grad_weight = grad_offset = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.autograd.grad(
                outputs=grad_output,
                inputs=input,
                grad_outputs=None,
                retain_graph=True,
                create_graph=True
            )[0]
        if ctx.needs_input_grad[1]:
            grad_weight = torch.autograd.grad(
                outputs=grad_output,
                inputs=weight,
                grad_outputs=None,
                retain_graph=True,
                create_graph=True
            )[0]
        if ctx.needs_input_grad[2]:
            grad_offset = torch.autograd.grad(
                outputs=grad_output,
                inputs=offset,
                grad_outputs=None,
                retain_graph=True,
                create_graph=True
            )[0]

        return (grad_input, grad_weight, grad_offset, None, None, None, None, None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(
            input, 
            weight, 
            bias, 
            ctx._bufs[0], 
            offset, 
            mask, 
            output,
            ctx._bufs[1], 
            weight.shape[2], 
            weight.shape[3], 
            ctx.stride,
            ctx.stride, 
            ctx.padding, 
            ctx.padding, 
            ctx.dilation, 
            ctx.dilation,
            ctx.groups, 
            ctx.deformable_groups, 
            ctx.with_bias
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(
            input, 
            weight, 
            bias, 
            ctx._bufs[0], 
            offset, 
            mask, 
            ctx._bufs[1],
            grad_input, 
            grad_weight, 
            grad_bias, 
            grad_offset, 
            grad_mask,
            grad_output, 
            weight.shape[2], 
            weight.shape[3], 
            ctx.stride,
            ctx.stride, 
            ctx.padding, 
            ctx.padding, 
            ctx.dilation, 
            ctx.dilation,
            ctx.groups, 
            ctx.deformable_groups, 
            ctx.with_bias
        )
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply

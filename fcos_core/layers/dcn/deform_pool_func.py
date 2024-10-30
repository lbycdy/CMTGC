import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from torchvision.ops import roi_align


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(
        ctx,
        data,
        rois,
        offset,
        spatial_scale,
        out_size,
        out_channels,
        no_trans,
        group_size=1,
        part_size=None,
        sample_per_part=4,
        trans_std=.0
    ):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        # 使用 torch 的 roi_align 作为替代
        output = roi_align(
            input=data,
            boxes=rois,
            output_size=(out_size, out_size),
            spatial_scale=spatial_scale,
            sampling_ratio=sample_per_part  # 类似 sample_per_part
        )

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = torch.ones_like(output)  # 占位符

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = torch.zeros_like(rois)  # 返回 RoI 的梯度
        grad_offset = torch.zeros_like(offset)

        # 手动实现反向传播逻辑
        grad_input = torch.autograd.grad(
            outputs=grad_output,
            inputs=data,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True
        )[0]

        # 如果有需要，您可以手动计算 grad_rois 的值
        # 这里我们简单地返回零梯度，但你可以根据需求进行修改

        return (grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None)


# 使用 DeformRoIPoolingFunction
deform_roi_pooling = DeformRoIPoolingFunction.apply

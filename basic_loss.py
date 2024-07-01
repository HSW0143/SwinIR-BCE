import torch
from torch import nn as nn
from torch.nn import functional as F

from archs.vgg_arch import VGGFeatureExtractor
from utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from torchvision.transforms import ToPILImage
from PIL import Image
import datetime
import os
i = 0
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # print("pred shape------",pred.shape)
        # print("target shape------", target.shape)
        global i
        i = i + 1
        if i % 300 ==0:
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            to_pil = ToPILImage()
            p0 = target[0]
            p1 = pred[0]
            plc0 = to_pil(p0)
            plc1 = to_pil(p1)
            save_directory = "/data1/hsw/results/generates1"
            save_path = os.path.join(save_directory, f'L1{formatted_time}HR.png')
            save_path1 = os.path.join(save_directory, f'L1{formatted_time}SR.png')
            plc0.save(save_path)
            plc1.save(save_path1)
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class bceloss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(bceloss, self).__init__()
        self.criterion = nn.BCELoss()
        self.loss_weight = loss_weight
        self.sigmoid = nn.Sigmoid()

    def decimal_to_binary(self, input):
        """将形状为 (b, c, h, w) 的张量中的每个十进制像素值转换为8位二进制表示，并提取每一位"""
        b, c, h, w = input.shape
        # 用于存储每个位的张量
        binary_tensors = torch.zeros(8, b, c, h, w, dtype=torch.float32, device=input.device)

        for i in range(8):
            # 提取第 i 位的值
            # print(f"Binary tensor {i} data type: {binary_tensors.dtype}, device: {binary_tensors.device}")
            binary_tensors[i] = ((input.to(torch.uint8) >> i) & 1)
            #print(binary_tensors[i].dtype)
        return binary_tensors

    def forward(self, pred, SR, target, **kwargs):
        # global i
        # i = i + 1
        # if i % 30 == 0:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        to_pil = ToPILImage()
        p0 = target[0]
        p1 = SR[0]
        plc0 = to_pil(p0)
        plc1 = to_pil(p1)
        save_directory = "/data1/hsw/results/generates2"
        save_path = os.path.join(save_directory, f'{formatted_time}HR.png')
        save_path1 = os.path.join(save_directory, f'{formatted_time}SR.png')
        plc0.save(save_path)
        plc1.save(save_path1)
        
        # b, c, h, w = target.shape
        target = target * 255
        for i in range(len(pred)):
            # print("-----------------------------------------------------------------------",i)
            pred[i] = self.sigmoid(pred[i])
            #print(pred[i].dtype)
        # print("ok2")
        list = self.decimal_to_binary(target)
        # print("ok3")
        loss = 0
        for i in range(8):
            b_loss = self.criterion(pred[i], list[i])#.permute(0, 1, 2, 3).reshape(b, c*h*w)
            loss = loss + b_loss
        return self.loss_weight*loss



@LOSS_REGISTRY.register()
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, pred1, pred2, target, **kwargs):
        global i
        i = i + 1
        # pred1: (C, B, 256, H, W)
        # target: (B, C, H, W)
        # x = target
        # print("target0 ------", target)
        # 重新调整 pred 的形状为 (B * C, H * W)
        B, C, H, W = target.size()
        # pred1 = pred1.permute(1, 0, 3, 4, 2).reshape(B * C * H * W, NUM)
        # print("pred ------",pred1)
        # 重新调整 target 的形状为 (B * C, H * W)
        if i % 300 == 0:
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            to_pil = ToPILImage()
            p0 = target[0]
            p1 = pred2[0]
            plc0 = to_pil(p0)
            plc1 = to_pil(p1)
            save_directory = "/data1/hsw/results/generates1"
            save_path = os.path.join(save_directory, f'{formatted_time}HR.png')
            save_path1 = os.path.join(save_directory, f'{formatted_time}SR.png')
            plc0.save(save_path)
            plc1.save(save_path1)
        target = target * 255
        target = target.permute(0, 1, 2, 3).reshape(B * C * H * W).long()
        # print("target ------",target)
        # 计算交叉熵损失
        loss1 = self.criterion(pred1, target)
        # 计算L1损失
        # loss2 = l1_loss(pred2, x)
        return self.loss_weight * loss1


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

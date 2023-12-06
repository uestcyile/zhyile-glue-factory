"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
   https://github.com/rpautrat/SuperPoint
   The implementation of this model and its trained weights are made
   available under the MIT license.
"""
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from ..base_model import BaseModel



def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    # keypoints N * 2
    # descriptors batch * 128 * 60 * 80
    # b, c, h, w = descriptors.shape
    b, c, h, w = torch.tensor(descriptors.shape).tolist()
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)

    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    # train
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)

    # onnx
    # descriptors = bilinear_grid_sample(descriptors, keypoints.view(b, 1, -1, 2), align_corners=True)
    # descriptors = descriptors.reshape(b, c, -1)
    # desc_norm_full= descriptors.norm(p=2, dim=1, keepdim=True)
    # descriptors = descriptors / desc_norm_full

    # test
    # for i in range(300):
    #   for j in range(128):
    #     print(str(descriptors[0, j, i].cpu().detach().numpy())+" ")
    #   xx = 0

    return descriptors.permute(0, 2 ,1).unsqueeze(0)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_w = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (width - border))
    mask_h = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (height - border))
    mask = mask_h & mask_w

    return keypoints[mask], scores[mask]

def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices].unsqueeze(0), scores.unsqueeze(0)

def spatial_nms(scores, nms_radius: int):
    """
    Fast Non-maximum suppression to remove nearby points
    scores: B,H,W
    """
    assert(nms_radius >= 0)
    assert(len(scores.shape)==3)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    scores = scores.unsqueeze(dim=1)
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    res = torch.where(max_mask, scores, zeros)
    return res.squeeze(dim=1)

def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def clean_checkpoint(ckpt):
    new_ckpt = {}
    for i,k in ckpt.items():
        if i[0:6] == "module":
            new_ckpt[i[7:]] = k
        else:
            new_ckpt[i] = k
    return new_ckpt

class DescriptorHead_bn_48_resrep(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grid_size, using_bn=True):
        super(DescriptorHead_bn_48_resrep, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        self.keyCnt = 196 # 128  196
        self.convDa = torch.nn.Conv2d(input_channel, self.keyCnt, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.convDb = torch.nn.Conv2d(self.keyCnt, output_channel, kernel_size=1, stride=1, padding=0)
        
        self.bnDa, self.bnDb = None, None
        if using_bn:
            self.bnDa = torch.nn.BatchNorm2d(self.keyCnt)
            self.bnDb = torch.nn.BatchNorm2d(output_channel)
    
    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnDa(self.relu(self.convDa(x)))
            out = self.bnDb(self.convDb(out))
        else:
            out = self.relu(self.convDa(x))
            out = self.convDb(out)
    
        # train glue or test
        desc = out
        # desc = F.interpolate(out, scale_factor=self.grid_size, mode='bilinear', align_corners=False)
        desc = F.normalize(desc, p=2, dim=1)  # normalize by channel
        return {'desc_raw': out, 'desc': desc}
        
        
class DetectorHead_bn_48_resrep(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(DetectorHead_bn_48_resrep, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.keyCnt = 196 # 128  196
        self.convPa = torch.nn.Conv2d(input_channel, self.keyCnt, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(self.keyCnt, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa, self.bnPb = None,None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(self.keyCnt)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out = self.bnPb(self.convPb(out))  #(B,65,H,W)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)
            
        # for test  onnx
        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]
        return {'logits': out, 'prob': prob}
        # # for train spt
        # return {'logits': out}


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU(inplace=True)
        # self.nonlinearity = nn.RReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGGBackbone_New(torch.nn.Module):
    
    def __init__(self, deploy=False, use_se=False, input_channel=1):
        super(RepVGGBackbone_New, self).__init__()
        
        self.deploy = deploy
        self.use_se = use_se
        
        self.in_planes = 32  # 32
        
        self.stage0 = RepVGGBlock(in_channels=input_channel, out_channels=self.in_planes, kernel_size=3, stride=1,
                                  padding=1, deploy=self.deploy, use_se=self.use_se)
        
        self.cur_layer_idx = 1
        
        self.stage1 = self._make_stage(32, 2, stride=2)  # 32 为out channels
        self.stage2 = self._make_stage(64, 2, stride=2)
        self.stage3 = self._make_stage(96, 4, stride=2)  # 4 5
        self.stage4 = self._make_stage(128, 2, stride=1)
    
    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = 1  # self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        feat_map = self.stage4(out)
        return feat_map
    

class RepSuperPointNet(BaseModel):
    """ Pytorch definition of SuperPoint Network. """
    
    def _init(self, conf):
        self.conf = SimpleNamespace(**conf)

        print("superpoint_48dim, conf: \n", self.conf)

        # super(RepSuperPointNet, self).__init__()
        self.nms = self.conf.nms
        self.det_thresh = self.conf.det_thresh
        self.remove_borders_number = self.conf.remove_borders_number
        self.max_keypoints = self.conf.max_num_keypoints
        self.backbone = RepVGGBackbone_New(deploy=False, use_se=False, input_channel=self.conf.input_channel)
        ##
        self.detector_head = DetectorHead_bn_48_resrep(input_channel=self.conf.det_head.feat_in_dim,
                                          grid_size=self.conf.grid_size, using_bn=self.conf.using_bn)
        
        self.descriptor_head = DescriptorHead_bn_48_resrep(input_channel=self.conf.des_head.feat_in_dim,
                                                     output_channel=self.conf.des_head.feat_out_dim,
                                                     grid_size=self.conf.grid_size, using_bn=self.conf.using_bn)
        # load state dict
        restore_dict = torch.load(self.conf.model_state_file, map_location='cpu')
        self.load_state_dict(clean_checkpoint(restore_dict['model'] if 'model' in restore_dict else restore_dict))
        
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()        
        print("load state dict succeed for superpoint_48dim.")
    
    def _forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        image = x["image"] # BxCxHxW
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        feat_map = self.backbone(image)
        det_outputs = self.detector_head(feat_map)
        
        # for train light-glue
        scores = det_outputs['prob']
        scores = spatial_nms(scores, self.nms)
        b, h, w = torch.tensor(scores.shape).tolist()
        
        keypoints = [
            torch.nonzero(s > self.det_thresh)
            for s in scores]
        
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.remove_borders_number, h*8, w*8)
            for k, s in zip(keypoints, scores)]))
        # Keep the k keypoints with highest score
        # if curr_max_kp >= 0:
        keypoints, scores = list(zip(*[
            top_k_keypoints(k, s, self.max_keypoints)
            for k, s in zip(keypoints, scores)]))
        
        # Compute the dense descriptors &&Extract descriptors
        desc_outputs = self.descriptor_head(feat_map)
        descriptors = desc_outputs['desc']
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        return {
            'keypoints': torch.cat(keypoints, dim=0),
            'scores': torch.cat(scores, dim=0),
            'descriptors': torch.cat(descriptors, dim=0),
        }

    def loss(self, pred, data):
        raise NotImplementedError
        
    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            #     torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                




# class SuperPoint(BaseModel):
#     default_conf = {
#         "descriptor_dim": 256,
#         "nms_radius": 4,
#         "max_num_keypoints": None,
#         "force_num_keypoints": False,
#         "detection_threshold": 0.005,
#         "remove_borders": 4,
#         "descriptor_dim": 256,
#         "channels": [64, 64, 128, 128, 256],
#         "dense_outputs": None,
#     }

#     checkpoint_url = "https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth"  # noqa: E501

#     def _init(self, conf):
#         self.conf = SimpleNamespace(**conf)
#         self.stride = 2 ** (len(self.conf.channels) - 2)
#         channels = [1, *self.conf.channels[:-1]]

#         backbone = []
#         for i, c in enumerate(channels[1:], 1):
#             layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
#             if i < len(channels) - 1:
#                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             backbone.append(nn.Sequential(*layers))
#         self.backbone = nn.Sequential(*backbone)

#         c = self.conf.channels[-1]
#         self.detector = nn.Sequential(
#             VGGBlock(channels[-1], c, 3),
#             VGGBlock(c, self.stride**2 + 1, 1, relu=False),
#         )
#         self.descriptor = nn.Sequential(
#             VGGBlock(channels[-1], c, 3),
#             VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
#         )

#         state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url)
#         self.load_state_dict(state_dict)

#     def _forward(self, data):
#         image = data["image"] # BxCxHxW
#         if image.shape[1] == 3:  # RGB
#             scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
#             image = (image * scale).sum(1, keepdim=True)
#         features = self.backbone(image)
#         descriptors_dense = torch.nn.functional.normalize(
#             self.descriptor(features), p=2, dim=1
#         )

#         # Decode the detection scores
#         scores = self.detector(features)
#         scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
#         b, _, h, w = scores.shape
#         scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
#         scores = scores.permute(0, 1, 3, 2, 4).reshape(
#             b, h * self.stride, w * self.stride
#         )
#         scores = batched_nms(scores, self.conf.nms_radius)

#         # Discard keypoints near the image borders
#         if self.conf.remove_borders:
#             pad = self.conf.remove_borders
#             scores[:, :pad] = -1
#             scores[:, :, :pad] = -1
#             scores[:, -pad:] = -1
#             scores[:, :, -pad:] = -1

#         # Extract keypoints
#         if b > 1:
#             idxs = torch.where(scores > self.conf.detection_threshold)
#             mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
#         else:  # Faster shortcut
#             scores = scores.squeeze(0)
#             idxs = torch.where(scores > self.conf.detection_threshold)

#         # Convert (i, j) to (x, y)
#         keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
#         scores_all = scores[idxs]

#         keypoints = []
#         scores = []
#         for i in range(b):
#             if b > 1:
#                 k = keypoints_all[mask[i]]
#                 s = scores_all[mask[i]]
#             else:
#                 k = keypoints_all
#                 s = scores_all
#             if self.conf.max_num_keypoints is not None:
#                 k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)

#             keypoints.append(k)
#             scores.append(s)

#         if self.conf.force_num_keypoints:
#             keypoints = pad_and_stack(
#                 keypoints,
#                 self.conf.max_num_keypoints,
#                 -2,
#                 mode="random_c",
#                 bounds=(
#                     0,
#                     data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
#                 ),
#             )
#             scores = pad_and_stack(
#                 scores, self.conf.max_num_keypoints, -1, mode="zeros"
#             )
#         else:
#             keypoints = torch.stack(keypoints, 0)
#             scores = torch.stack(scores, 0)

#         if len(keypoints) == 1 or self.conf.force_num_keypoints:
#             # Batch sampling of the descriptors
#             desc = sample_descriptors(keypoints, descriptors_dense, self.stride)
#         else:
#             desc = [
#                 sample_descriptors(k[None], d[None], self.stride)[0]
#                 for k, d in zip(keypoints, descriptors_dense)
#             ]

#         pred = {
#             "keypoints": keypoints + 0.5,
#             "keypoint_scores": scores,
#             "descriptors": desc.transpose(-1, -2),
#         }
#         if self.conf.dense_outputs:
#             pred["dense_descriptors"] = descriptors_dense

#         return pred

#     def loss(self, pred, data):
#         raise NotImplementedError

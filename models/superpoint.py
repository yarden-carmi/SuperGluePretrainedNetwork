# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

try:
    from runspace.src.ops.observed_ops import (
        ObservedDiscardTrash,
        ObservedReorderReshape,
        ObservedSimpleNMS,
        ObservedThreshold,
        ObservedRemoveBorders,
        ObservedTopKKeypoints,
        ObservedKeypointFlip,
        ObservedL2Norm,
        ObservedCoordOps,
        ObservedGridSample,
    )
except ImportError:
    import torch.nn.functional as _F

    class ObservedDiscardTrash(nn.Module):
        def forward(self, x): return x[:, :-1]

    class ObservedReorderReshape(nn.Module):
        def forward(self, scores):
            b, _, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            return scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

    class ObservedSimpleNMS(nn.Module):
        def __init__(self, radius=4):
            super().__init__()
            self.radius = radius
        def forward(self, scores):
            return simple_nms(scores, self.radius)

    class ObservedThreshold(nn.Module):
        def __init__(self, threshold=0.005):
            super().__init__()
            self.threshold = threshold
        def forward(self, scores):
            keypoints = [torch.nonzero(s > self.threshold) for s in scores]
            score_list = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
            return keypoints, score_list

    class ObservedRemoveBorders(nn.Module):
        def __init__(self, border=4):
            super().__init__()
            self.border = border
        def forward(self, keypoints, scores, h_full, w_full):
            result = [remove_borders(k, s, self.border, h_full, w_full)
                      for k, s in zip(keypoints, scores)]
            return list(zip(*result))

    class ObservedTopKKeypoints(nn.Module):
        def __init__(self, max_keypoints=-1):
            super().__init__()
            self.max_keypoints = max_keypoints
        def forward(self, keypoints, scores):
            if self.max_keypoints < 0:
                return list(keypoints), list(scores)
            result = [top_k_keypoints(k, s, self.max_keypoints)
                      for k, s in zip(keypoints, scores)]
            return list(zip(*result))

    class ObservedKeypointFlip(nn.Module):
        def forward(self, keypoints):
            return [torch.flip(k, [1]).float() for k in keypoints]

    class ObservedL2Norm(nn.Module):
        def forward(self, x):
            return _F.normalize(x, p=2, dim=1)

    class ObservedCoordOps(nn.Module):
        def __init__(self, s=8):
            super().__init__()
            self.s = s
        def forward(self, keypoints, descriptors):
            _, _, h, w = descriptors.shape
            s = self.s
            kp = keypoints - s / 2 + 0.5
            kp = kp / torch.tensor(
                [(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                dtype=kp.dtype, device=kp.device)[None]
            return kp * 2 - 1

    class ObservedGridSample(nn.Module):
        def forward(self, descriptors, keypoints_norm):
            b, c, _, _ = descriptors.shape
            args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
            desc = _F.grid_sample(
                descriptors, keypoints_norm.view(b, 1, -1, 2),
                mode='bilinear', **args)
            return _F.normalize(desc.reshape(b, c, -1), p=2, dim=1)


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        # --- Phase 1: per-call ReLU instances (one per conv output) ---
        self.relu_conv1a = nn.ReLU(inplace=True)   # stageA1
        self.relu_conv1b = nn.ReLU(inplace=True)   # stageA2
        self.relu_conv2a = nn.ReLU(inplace=True)   # stageA4
        self.relu_conv2b = nn.ReLU(inplace=True)   # stageA5
        self.relu_conv3a = nn.ReLU(inplace=True)   # stageA7
        self.relu_conv3b = nn.ReLU(inplace=True)   # stageA8
        self.relu_conv4a = nn.ReLU(inplace=True)   # stageA10
        self.relu_conv4b = nn.ReLU(inplace=True)   # stageA11
        self.relu_convPa = nn.ReLU(inplace=True)   # stageB1
        self.relu_convDa = nn.ReLU(inplace=True)   # stageC1

        # --- Phase 1: per-call MaxPool instances ---
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # stageA3
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # stageA6
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # stageA9

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        # --- Phase 2: Softmax as registered module (stageB2) ---
        self.softmax = nn.Softmax(dim=1)

        # --- Phase 3/4: Observed post-processing modules ---
        self.discard_trash   = ObservedDiscardTrash()                               # stageB3
        self.reorder_reshape = ObservedReorderReshape()                             # stageB4-B7
        self.nms             = ObservedSimpleNMS(radius=self.config['nms_radius'])  # stageB8
        self.threshold_op    = ObservedThreshold(
            threshold=self.config['keypoint_threshold'])                            # stageB10-B12
        self.remove_borders_op = ObservedRemoveBorders(
            border=self.config['remove_borders'])                                   # stageB9
        self.top_k_op        = ObservedTopKKeypoints(
            max_keypoints=self.config['max_keypoints'])                             # stageB13
        self.keypoint_flip   = ObservedKeypointFlip()                               # stageB14

        self.l2_norm         = ObservedL2Norm()                                     # stageC3
        self.coord_ops       = ObservedCoordOps(s=8)                                # stageD1-D3
        self.grid_sample_op  = ObservedGridSample()                                 # stageD4

        # strict=False: new modules (relu_*, pool*, softmax, observed ops) have no saved state
        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # --- Shared Encoder ---
        x = self.relu_conv1a(self.conv1a(data['image']))   # stageA1
        x = self.relu_conv1b(self.conv1b(x))               # stageA2
        x = self.pool1(x)                                  # stageA3
        x = self.relu_conv2a(self.conv2a(x))               # stageA4
        x = self.relu_conv2b(self.conv2b(x))               # stageA5
        x = self.pool2(x)                                  # stageA6
        x = self.relu_conv3a(self.conv3a(x))               # stageA7
        x = self.relu_conv3b(self.conv3b(x))               # stageA8
        x = self.pool3(x)                                  # stageA9
        x = self.relu_conv4a(self.conv4a(x))               # stageA10
        x = self.relu_conv4b(self.conv4b(x))               # stageA11

        # --- Detector head ---
        cPa    = self.relu_convPa(self.convPa(x))          # stageB1
        scores = self.convPb(cPa)                          # stageB2 (conv)
        scores = self.softmax(scores)                      # stageB2 (softmax)
        scores = self.discard_trash(scores)                # stageB3: drop dustbin channel
        _, _, h, w = scores.shape                          # h=H/8, w=W/8
        scores = self.reorder_reshape(scores)              # stageB4-B7: pixel shuffle
        scores = self.nms(scores)                          # stageB8: NMS

        # stageB10-B12: threshold + extract keypoints and scores
        keypoints, score_list = self.threshold_op(scores)
        # stageB9: remove border keypoints (applied after extraction)
        keypoints, score_list = self.remove_borders_op(keypoints, score_list, h*8, w*8)
        # stageB13: keep top-k
        keypoints, score_list = self.top_k_op(keypoints, score_list)
        # stageB14: flip (h,w) -> (x,y)
        keypoints = self.keypoint_flip(keypoints)

        # --- Descriptor head ---
        cDa         = self.relu_convDa(self.convDa(x))    # stageC1
        descriptors = self.convDb(cDa)                    # stageC2
        descriptors = self.l2_norm(descriptors)           # stageC3: dense L2 norm

        # --- Descriptor sampling (stageD) ---
        descriptors = [
            self.grid_sample_op(
                d[None],
                self.coord_ops(k[None], d[None])          # stageD1-D3: coord ops
            )[0]                                          # stageD4: bilinear grid_sample + L2
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            'keypoints': keypoints,
            'scores': score_list,
            'descriptors': descriptors,
        }

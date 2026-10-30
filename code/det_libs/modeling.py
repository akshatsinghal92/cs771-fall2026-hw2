import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms, remove_small_boxes

import torch
from torch import nn
from torch.nn import functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        The head will be applied to all levels of the feature pyramid, and predict 
        a single logit for each location on every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        """
        outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            out = self.cls_logits(out)
            outputs += (out, )
        return outputs


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The logic is rather similar to FCOSClassificationHead. The key difference is 
        that this head bundles both regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        """
        reg_outputs = tuple()
        ctr_outputs = tuple()
        for feats in x:
            out = self.conv(feats)
            reg_out = self.bbox_reg(out)
            ctr_out = self.bbox_ctrness(out)
            reg_outputs += (reg_out, )
            ctr_outputs += (ctr_out, )
        return reg_outputs, ctr_outputs


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FOCS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # fix all bn ops, yet still allow the update of affine params
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

#########################################################################################
# Part II-2: FCOS training
#########################################################################################
    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        device = cls_logits[0].device
        num_classes = self.num_classes

        # ---- flatten predictions across pyramid levels: (N, P, .) ----
        cls_logits_flat = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, num_classes) for x in cls_logits],
            dim=1,
        )  # (N, P, C)
        reg_flat = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4) for x in reg_outputs], dim=1
        )  # (N, P, 4)  -- predicted (l, t, r, b) in feature-grid units (>= 0)
        ctr_flat = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1) for x in ctr_logits], dim=1
        )  # (N, P)

        # ---- per-point geometry, shared across images in the batch ----
        points_all = torch.cat([p.reshape(-1, 2) for p in points], dim=0)  # (P, 2) = (y, x)
        strides_all = torch.cat(
            [torch.full((p.shape[0] * p.shape[1],), float(s), device=device)
             for p, s in zip(points, strides)]
        )  # (P,)
        reg_range_all = torch.cat(
            [rr.view(1, 2).expand(p.shape[0] * p.shape[1], 2)
             for p, rr in zip(points, reg_range)], dim=0
        )  # (P, 2) -- in pixels

        py, px = points_all[:, 0], points_all[:, 1]  # vertical (y), horizontal (x)
        P = points_all.shape[0]
        INF = 1e8

        cls_targets = torch.zeros_like(cls_logits_flat)  # (N, P, C)
        pred_boxes_list, gt_boxes_list = [], []
        ctr_pred_list, ctr_tgt_list = [], []
        num_pos = 0

        # ---- assign targets per image (vectorized over points x GT boxes) ----
        for i, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)    # (M, 4) in (x1, y1, x2, y2)
            gt_labels = target["labels"].to(device)  # (M,) already 0-indexed
            M = gt_boxes.shape[0]
            if M == 0:
                continue

            x1, y1, x2, y2 = gt_boxes.unbind(dim=1)  # each (M,)

            # distances from every point to every box's sides -> (P, M, 4)
            l = px[:, None] - x1[None, :]
            t = py[:, None] - y1[None, :]
            r = x2[None, :] - px[:, None]
            b = y2[None, :] - py[:, None]
            reg_targets = torch.stack([l, t, r, b], dim=-1)  # (P, M, 4)

            # center sampling: a radius*stride window clamped to the box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            radius = self.center_sampling_radius * strides_all[:, None]  # (P, 1)
            xmin = torch.maximum(x1[None, :], cx[None, :] - radius)
            xmax = torch.minimum(x2[None, :], cx[None, :] + radius)
            ymin = torch.maximum(y1[None, :], cy[None, :] - radius)
            ymax = torch.minimum(y2[None, :], cy[None, :] + radius)
            in_center = (
                (px[:, None] >= xmin) & (px[:, None] <= xmax)
                & (py[:, None] >= ymin) & (py[:, None] <= ymax)
            )  # (P, M)

            # each pyramid level only handles boxes within its regression range
            max_reg = reg_targets.max(dim=-1).values  # (P, M)
            in_range = (max_reg >= reg_range_all[:, 0:1]) & (max_reg <= reg_range_all[:, 1:2])

            valid = in_center & in_range & (reg_targets.min(dim=-1).values > 0)  # (P, M)

            # resolve points matching multiple GTs by picking the smallest-area box
            areas = (x2 - x1) * (y2 - y1)  # (M,)
            area_mat = areas[None, :].expand(P, M).clone()
            area_mat[~valid] = INF
            min_area, gt_idx = area_mat.min(dim=1)  # (P,)
            fg_idx = torch.nonzero(min_area < INF).squeeze(1)
            if fg_idx.numel() == 0:
                continue
            num_pos += fg_idx.numel()
            matched = gt_idx[fg_idx]  # (nfg,)

            # (1) classification target: one-hot of the matched class
            cls_targets[i, fg_idx, gt_labels[matched]] = 1.0

            # (2) regression: decode predicted boxes at fg points (grid units -> pixels)
            pred_ltrb = reg_flat[i][fg_idx] * strides_all[fg_idx][:, None]  # (nfg, 4)
            xf, yf = px[fg_idx], py[fg_idx]
            pred_boxes_list.append(
                torch.stack(
                    [xf - pred_ltrb[:, 0], yf - pred_ltrb[:, 1],
                     xf + pred_ltrb[:, 2], yf + pred_ltrb[:, 3]], dim=1
                )
            )
            gt_boxes_list.append(gt_boxes[matched])

            # (3) center-ness target derived from the regression target
            reg_t_pos = reg_targets[fg_idx, matched]  # (nfg, 4)
            lr, tb = reg_t_pos[:, [0, 2]], reg_t_pos[:, [1, 3]]
            ctrness = torch.sqrt(
                (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6))
                * (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
            )
            ctr_tgt_list.append(ctrness)
            ctr_pred_list.append(ctr_flat[i][fg_idx])

        # ---- normalize by #positives (EMA for small-batch stability) ----
        if self.training:
            self.loss_normalizer = (
                self.loss_normalizer_momentum * self.loss_normalizer
                + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
            )
        normalizer = max(self.loss_normalizer, 1.0)

        # classification loss is computed over ALL points (sigmoid focal)
        cls_loss = sigmoid_focal_loss(
            cls_logits_flat, cls_targets, reduction="sum"
        ) / normalizer

        # regression (GIoU) and center-ness (BCE) only over FOREGROUND points
        if len(pred_boxes_list) > 0:
            pred_boxes = torch.cat(pred_boxes_list, dim=0)
            gt_boxes_cat = torch.cat(gt_boxes_list, dim=0)
            reg_loss = giou_loss(pred_boxes, gt_boxes_cat, reduction="sum") / normalizer
            ctr_pred = torch.cat(ctr_pred_list, dim=0)
            ctr_tgt = torch.cat(ctr_tgt_list, dim=0)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred, ctr_tgt, reduction="sum"
            ) / normalizer
        else:
            # no positives in this batch: keep graph connected with zero losses
            reg_loss = reg_flat.sum() * 0.0
            ctr_loss = ctr_flat.sum() * 0.0

        final_loss = cls_loss + reg_loss + ctr_loss
        losses = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }
        return losses


#########################################################################################
# Part II-1: FCOS Inference
#########################################################################################
    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """
    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        num_images = cls_logits[0].shape[0]
        num_levels = len(cls_logits)
        detections = []

        # it is easier to reason about the inference one image at a time
        for img_idx in range(num_images):
            image_boxes = []
            image_scores = []
            image_labels = []

            # (a) loop over every pyramid level
            for level in range(num_levels):
                # predictions for this image / level
                cls_logit = cls_logits[level][img_idx]    # (C, H, W)
                reg_output = reg_outputs[level][img_idx]  # (4, H, W) -> (l, t, r, b)
                ctr_logit = ctr_logits[level][img_idx]    # (1, H, W)
                stride = strides[level]
                level_points = points[level]              # (H, W, 2) as (y, x)

                num_classes = cls_logit.shape[0]

                # flatten the spatial dims (row-major, consistent across tensors)
                cls_prob = cls_logit.permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
                ctr_prob = ctr_logit.permute(1, 2, 0).reshape(-1).sigmoid()
                reg = reg_output.permute(1, 2, 0).reshape(-1, 4)
                pts = level_points.reshape(-1, 2)

                # (1) object score per (location, class): sqrt(cls * center-ness)
                scores = torch.sqrt(cls_prob * ctr_prob[:, None]).flatten()

                # (2) keep candidates above the score threshold
                keep = scores > self.score_thresh
                scores = scores[keep]
                keep_idxs = torch.nonzero(keep).squeeze(1)

                # (3) keep only the top-K scoring candidates (before NMS)
                num_topk = min(self.topk_candidates, scores.shape[0])
                scores, topk = scores.topk(num_topk)
                keep_idxs = keep_idxs[topk]

                # recover the location index and the class index
                point_idxs = torch.div(keep_idxs, num_classes, rounding_mode="floor")
                class_idxs = keep_idxs % num_classes

                # (4) decode boxes. The regression targets are in feature-grid
                # units (divided by the stride in training), so scale back up here.
                sel_reg = reg[point_idxs] * stride         # (K, 4)
                sel_pts = pts[point_idxs]                  # (K, 2) as (y, x)
                y_ctr, x_ctr = sel_pts[:, 0], sel_pts[:, 1]
                left, top, right, bottom = sel_reg.unbind(dim=1)
                boxes = torch.stack(
                    [x_ctr - left, y_ctr - top, x_ctr + right, y_ctr + bottom],
                    dim=1,
                )                                          # (K, 4) in (x1, y1, x2, y2)

                # (5) clip boxes to the image boundary (padding is outside this)
                img_h, img_w = image_shapes[img_idx]
                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=img_w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=img_h)

                image_boxes.append(boxes)
                image_scores.append(scores)
                image_labels.append(class_idxs)

            # (b) collect candidates across all pyramid levels
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # drop degenerate / tiny boxes created by clipping
            keep = remove_small_boxes(image_boxes, min_size=1e-2)
            image_boxes = image_boxes[keep]
            image_scores = image_scores[keep]
            image_labels = image_labels[keep]

            # (c) per-class NMS (boxes of different categories may overlap)
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            # (d) keep the top detections after NMS
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    # +1 to compensate for the input label transform (0 = background)
                    "labels": image_labels[keep] + 1,
                }
            )

        return detections
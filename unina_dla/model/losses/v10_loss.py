import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

class v10DetectionLoss(v8DetectionLoss):
    """
    YOLOv10 Loss Function with One-to-One (O2O) Assignment support.
    
    Inherits from v8DetectionLoss but modifies assignment strategy to support
    NMS-free training by enforcing strict one-to-one matching during the
    dual assignment phase (if enabled) or as the primary assignment.
    """
    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        # One-to-One assigner uses topk=1 to ensure unique best match
        self.assigner_o2o = TaskAlignedAssigner(topk=1, num_classes=self.nc, alpha=0.5, beta=6.0)

    def __call__(self, preds, batch):
        """Calculate loss."""
        # preds: tuple(reg, cls)
        # but standard v8 loss expects [B, 4+NC, A]
        # We need to adapt input if it comes from our UNINA_DLA which returns tuple
        
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # ... logic to extract feats ...
        # Assume feats is the list of concatenated [reg+cls] per scale
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy

        # Assignment
        # One-to-Many (Standard v8/v10 auxiliary)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        # One-to-One (NMS-Free primary)
        # Using TAL with topk=1 is a strong approximation of Hungarian for YOLOv10
        _, target_bboxes_o2o, target_scores_o2o, fg_mask_o2o, _ = self.assigner_o2o(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
            
        # We compute loss on O2O targets primarily for the NMS-free head
        # In a full v10 implementation, we would sum both losses.
        # For UNINA_DLA simplified student, we focus on the O2O signal to ensure NMS-free capability.
        
        target_scores_sum = torch.max(target_scores_o2o, target_scores) # Combine for stability or just use O2O?
        # Let's use strict O2O for the main branch to enforce sparseness
        target_scores = target_scores_o2o
        target_bboxes = target_bboxes_o2o
        fg_mask = fg_mask_o2o

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()

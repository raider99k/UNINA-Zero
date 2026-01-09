
import torch
from unina_dla.utils.box_ops import decode_bboxes, non_max_suppression
from unina_dla.utils.metrics import UNINAMetrics
from tqdm import tqdm

def validate(model, dataloader, device, conf_thres=0.001, iou_thres=0.6, strides=[8, 16, 32], num_classes=1, names=None):
    model.eval()
    
    metrics = UNINAMetrics(num_classes=num_classes, names=names)
    
    # Validation Loop
    # Suppress tqdm if inside training? maybe pass pbar
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    for batch in pbar:
        imgs = batch['img'].to(device).float() / 255.0
        bs = imgs.shape[0]
        
        # Process Targets
        targets = []
        if 'bboxes' in batch:
            b_idx = batch['batch_idx'].to(device)
            b_cls = batch['cls'].to(device).view(-1, 1)
            b_box = batch['bboxes'].to(device)
            
            # Convert xywh normalized to xyxy absolute
            h, w = imgs.shape[2:]
            b_box_abs = b_box.clone()
            b_box_abs[:, 0] *= w
            b_box_abs[:, 1] *= h
            b_box_abs[:, 2] *= w
            b_box_abs[:, 3] *= h
            
            x1 = b_box_abs[:, 0] - b_box_abs[:, 2] / 2
            y1 = b_box_abs[:, 1] - b_box_abs[:, 3] / 2
            x2 = b_box_abs[:, 0] + b_box_abs[:, 2] / 2
            y2 = b_box_abs[:, 1] + b_box_abs[:, 3] / 2
            
            b_box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            gt_data = torch.cat([b_cls, b_box_xyxy], dim=1)
            
            for i in range(bs):
                mask = (b_idx == i)
                if mask.sum() > 0:
                    targets.append(gt_data[mask])
                else:
                    targets.append(torch.zeros((0, 5), device=device))
        else:
             targets = [torch.zeros((0, 5), device=device) for _ in range(bs)]
             
        # Inference
        with torch.no_grad():
            preds_reg, preds_cls = model(imgs) 
            
            # Anchors
            anchors_list = []
            strides_list = []
            for i, stride in enumerate(strides):
                 # preds_cls[i] is [B, nc, H, W]
                 B, C, H, W = preds_cls[i].shape
                 sy, sx = torch.meshgrid(
                     torch.arange(H, device=device) + 0.5,
                     torch.arange(W, device=device) + 0.5,
                     indexing='ij'
                 )
                 anchors = torch.stack((sx, sy), -1).reshape(-1, 2) * stride
                 anchors_list.append(anchors)
                 strides_list.append(torch.full((H*W, 1), stride, device=device))
                 
            all_anchors = torch.cat(anchors_list)
            all_strides = torch.cat(strides_list)
            
            # Process Outputs
            pred_cls = []
            pred_reg = []
            
            for i in range(len(strides)):
                 c = preds_cls[i].permute(0, 2, 3, 1).reshape(bs, -1, num_classes).sigmoid()
                 pred_cls.append(c)
                 r = preds_reg[i].permute(0, 2, 3, 1).reshape(bs, -1, 4 * 16) # reg_max=16 hardcoded or pass it
                 pred_reg.append(r)
                 
            pred_cls = torch.cat(pred_cls, 1)
            pred_reg = torch.cat(pred_reg, 1)
            
            pred_boxes = decode_bboxes(pred_reg, all_anchors, all_strides, xywh=False)
            
            # NMS Prep
            # Objectness is max class score here (since no obj head)
            conf, _ = pred_cls.max(dim=-1, keepdim=True)
            
            nms_input = torch.cat([pred_boxes, conf, pred_cls], dim=-1)
            
            dets = non_max_suppression(nms_input, conf_thres=conf_thres, iou_thres=iou_thres)
            
            metrics.update(dets, targets)
            
    return metrics.compute()

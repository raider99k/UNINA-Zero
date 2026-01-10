import torch
import torchvision

def dfl_integral(dfl_preds, reg_max=16):
    """
    Decodes DFL distribution into scalar coordinates.
    Assumes dfl_preds are unnormalized logits, and applies softmax.

    Args:
        dfl_preds: [B, 4*reg_max, H, W] or [B, N, 4*reg_max] (logits)
        reg_max: number of bins
    Returns:
        coords: [B, 4, H, W] or [B, N, 4]
    """
    # Assuming dfl_preds is logits, we need softmax
    # Shape handling
    if dfl_preds.dim() == 4:
        B, C, H, W = dfl_preds.shape
        # [B, 4, 16, H, W]
        dfl_preds = dfl_preds.view(B, 4, reg_max, H, W)
        dfl_preds = dfl_preds.softmax(dim=2)
        # Weight [0, 1, ..., 15]
        weight = torch.arange(reg_max, device=dfl_preds.device).view(1, 1, -1, 1, 1).float()
        # Sum(P * W)
        return (dfl_preds * weight).sum(dim=2) # [B, 4, H, W]
    else:
        # Flattened case [B, N, 4*reg_max]
        B, N, C = dfl_preds.shape
        dfl_preds = dfl_preds.view(B, N, 4, reg_max)
        dfl_preds = dfl_preds.softmax(dim=3)
        weight = torch.arange(reg_max, device=dfl_preds.device).view(1, 1, 1, -1).float()
        return (dfl_preds * weight).sum(dim=3) # [B, N, 4]

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchors from features.
    Args:
        feats: list of features
        strides: list of strides
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def decode_bboxes(dfl_preds, anchors, strides, xywh=False):
    """
    Decode DFL predictions to bounding boxes.
    dfl_preds: [B, N, 4*reg_max] (flattened) or [B, N, 4] (if already integrated)
    anchors: [N, 2]
    strides: [N, 1]
    """
    if dfl_preds.shape[-1] > 4:
        # Assuming reg_max=16
        tlrb = dfl_integral(dfl_preds, reg_max=dfl_preds.shape[-1]//4)
    else:
        tlrb = dfl_preds
        
    tlrb = tlrb * strides
    boxes = dist2bbox(tlrb, anchors, xywh=xywh)
    return boxes

def xywh2xyxy(x):
    """
    Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2].
    x: [..., 4]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, xywh=False):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    prediction: [B, N, 4+C] or [B, N, 4+reg_max+C]? 
                Assuming prediction is [B, N, 4+C] where 4 is xywh or xyxy
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'

    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        # CRITICAL: Removed redundant multiplication for Anchor-Free (No Objectness)
        # x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]
        # if box is xywh, convert to xyxy
        if xywh:
            box = xywh2xyxy(box)
        
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_det:
            x = x[x[:, 4].argsort(descending=True)[:max_det]]  # sort by confidence
            
        # Batched NMS
        c = x[:, 5:6] * 7680  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        
    return output

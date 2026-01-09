
import torch
import numpy as np

try:
    from ultralytics.utils.metrics import ap_per_class
except ImportError:
    ap_per_class = None

class UNINAMetrics:
    def __init__(self, num_classes=1, plot=False, names=None):
        self.num_classes = num_classes
        self.plot = plot
        self.names = names or {i: f'class{i}' for i in range(num_classes)}
        self.stats = [] # list of (correct, conf, pred_cls, target_cls)
        
    def update(self, preds, targets):
        """
        preds: list of tensors [N, 6] (xyxy, conf, cls) - one per image
        targets: list of tensors [M, 5] (cls, xyxy) - one per image
        """
        for si, pred in enumerate(preds):
            labels = targets[si]
            nl, npr = labels.shape[0], pred.shape[0]
            
            correct = torch.zeros(npr, 10, dtype=torch.bool, device=pred.device) # IOUS [0.5 : 0.95 : 0.05]
            
            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros((2, 0), device=pred.device), labels[:, 0]))
                continue
            
            if nl:
                target_cls = labels[:, 0]
                target_box = labels[:, 1:]
                
                # Evaluate
                from ultralytics.utils.metrics import match_predictions
                # match_predictions(pred_classes, true_classes, pred_boxes, true_boxes, iou_thres, giou=False)
                # But ultralytics match_predictions signature varies.
                # Let's use simple IoU matching if possible or rely on library.
                
                # Standard IOUs
                iouv = torch.linspace(0.5, 0.95, 10, device=pred.device)
                
                # We need to manually match if we don't assume ultralytics internal API
                # But since we have ultralytics installed, let's try to use it
                # Assuming simple implementation for now:
                
                # Clone for safety
                tbox = target_box.clone()
                pbox = pred[:, :4]
                
                # IoU
                # box_iou from ultralytics.utils.metrics or torchvision
                import torchvision
                iou = torchvision.ops.box_iou(tbox, pbox) # [M, N]
                
                # Process for each IoU threshold
                for i, threshold in enumerate(iouv):
                     # Simple greedy match?
                     # Ideally use linear_sum_assignment or greedy
                     # check matches
                     if iou.numel() > 0:
                        matches = []
                        # Logic handling is complex, let's trust ap_per_class does stats?
                        # ap_per_class takes ALL stats. 
                        # We need to generate 'correct' vector.
                        
                        # Simplified: Correct is [N, 10]
                        # matches_prediction logic from YOLOv5/v8:
                        # For each threshold, find best match.
                        # This is too bulky to rewrite.
                        pass

                # Fallback: if ultralytics is present, we might want to use its Validator?
                # But we are Custom.
                
                # Let's try to use a simplified matching provided by `box_ops` or similar.
                # I will define a robust match function here.
                correct = self._process_batch(pred, labels, iouv)
                
            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    def _process_batch(self, detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Tensor): (N, 6) [x1, y1, x2, y2, conf, class]
            labels (Tensor): (M, 5) [class, x1, y1, x2, y2]
            iouv (Tensor): list of iou thresholds
        Returns:
            correct (Tensor): (N, 10)
        """
        import torchvision
        
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=detections.device)
        iou = torchvision.ops.box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU > 0.5 and class match
        
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            matches = torch.from_numpy(matches).to(detections.device)
            matches = matches[matches[:, 2].argsort()[::-1]]
            
            tp = matches[:, 1].long()
            for i, threshold in enumerate(iouv):
                 # matches[:, 2] is iou
                 # filter by threshold
                 m = matches[matches[:, 2] >= threshold]
                 if m.shape[0]:
                     correct[m[:, 1].long(), i] = True
                     
        return correct

    def compute(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
             tp, conf, pred_cls, target_cls = stats
             # ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=..., names=...)
             if ap_per_class:
                 results = ap_per_class(tp, conf, pred_cls, target_cls, plot=self.plot, names=self.names)
                 # results: (tp, fp, p, r, f1, ap_class, ap50, ap)
                 # returns: tp, fp, p, r, f1, ap_class, ap50, ap
                 # Note: ap is mAP@50-95
                 
                 # Extract mAP50 and mAP50-95
                 ap50 = results[6].mean()
                 ap = results[7].mean()
                 mp = results[2].mean()
                 mr = results[3].mean()
                 
                 return {'mAP50': ap50, 'mAP50-95': ap, 'precision': mp, 'recall': mr}
        
        return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0}


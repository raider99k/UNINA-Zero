
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ultralytics.utils.metrics import ap_per_class
except ImportError:
    ap_per_class = None

class UNINAMetrics:
    def __init__(self, num_classes=1, plot=False, names=None, save_dir=None):
        self.num_classes = num_classes
        self.plot = plot
        self.save_dir = save_dir
        self.names = names or {i: f'class{i}' for i in range(num_classes)}
        self.stats = [] # list of (correct, conf, pred_cls, target_cls)
        # Confusion matrix: rows=GT, cols=Pred, +1 for background
        self.confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        
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
                # Evaluate using custom matching logic
                # We implement a greedy matching strategy in _process_batch which is distinct
                # from the complex assignment often used, but effective and faster.
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
        Also updates the confusion matrix.
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
        
        matched_gt = set()
        matched_pred = set()
        
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
            # Correct sorting: matches are (gt_idx, pred_idx, iou)
            # We must prioritize high-confidence predictions first to prevent matching theft.
            # Get confidence for each prediction index in matches
            confidences = detections[matches[:, 1].astype(int), 4].cpu().numpy()
            
            # Sort by Confidence (desc) then IoU (desc)
            # lexicographical sort in numpy: sort by iou, then by confidence
            # np.lexsort takes (keys to sort by) -> first key is primary sort (low to high)
            # We want high to low, so we use negative.
            sort_indices = np.lexsort((-matches[:, 2], -confidences))
            matches = matches[sort_indices]

            # Greedy match: highest rank (conf then iou) gets priority
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            matches = torch.from_numpy(matches).to(detections.device)
            
            # Update confusion matrix for matched predictions
            for m in matches:
                gt_idx, pred_idx, iou_val = int(m[0]), int(m[1]), float(m[2])
                if iou_val >= 0.5:  # Use IoU@0.5 for confusion matrix
                    gt_cls = int(labels[gt_idx, 0])
                    pred_cls = int(detections[pred_idx, 5])
                    self.confusion_matrix[gt_cls, pred_cls] += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
            
            tp = matches[:, 1].long()
            for i, threshold in enumerate(iouv):
                 # matches[:, 2] is iou
                 # filter by threshold
                 m = matches[matches[:, 2] >= threshold]
                 if m.shape[0]:
                     correct[m[:, 1].long(), i] = True
        
        # False negatives: GT boxes not matched
        for gt_idx in range(labels.shape[0]):
            if gt_idx not in matched_gt:
                gt_cls = int(labels[gt_idx, 0])
                self.confusion_matrix[gt_cls, self.num_classes] += 1  # Predicted as background
        
        # False positives: Predictions not matched
        for pred_idx in range(detections.shape[0]):
            if pred_idx not in matched_pred:
                pred_cls = int(detections[pred_idx, 5])
                self.confusion_matrix[self.num_classes, pred_cls] += 1  # GT was background
                     
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

    def plot_confusion_matrix(self, filename=None):
        """Plot and save the confusion matrix."""
        labels = [self.names.get(i, f'class{i}') for i in range(self.num_classes)] + ['background']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix')
        
        save_path = filename or (self.save_dir + '/confusion_matrix.png' if self.save_dir else 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
        return save_path

import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFDistillationLoss(nn.Module):
    """
    Vector 1: Scale-Decoupled Feature (SDF) Distillation.
    Includes spatial attention mask to focus on foreground objects.
    """
    def __init__(self, mask_type='soft', threshold=0.4):
        super().__init__()
        self.mask_type = mask_type
        self.threshold = threshold
        
    def generate_attention_mask(self, teacher_feats):
        """
        Generate binary mask from teacher features based on magnitude.
        Mask = 1 where mean(abs(feature)) > threshold
        """
        masks = []
        for feat in teacher_feats:
            # simple magnitude attention
            # feat: [B, C, H, W]
            magnitude = torch.mean(torch.abs(feat), dim=1, keepdim=True) # [B, 1, H, W]
            
            # Robust Normalization (avoid outliers)
            # Use 5th and 95th percentile to clamp
            # Note: quantile might be slow on large batches/CPUs, but fine for GPU training
            
            # Flatten spatial dims for quantile
            b, c, h, w = magnitude.shape
            flat_mag = magnitude.view(b, c, -1)
            
            m_min = torch.quantile(flat_mag, 0.05, dim=2, keepdim=True).unsqueeze(-1)
            m_max = torch.quantile(flat_mag, 0.95, dim=2, keepdim=True).unsqueeze(-1)
            
            # Clamp and Normalize
            # Broadcast back to [B, 1, H, W]
            magnitude_clamped = magnitude.clamp(min=m_min, max=m_max)
            diff = m_max - m_min
            norm_mag = (magnitude_clamped - m_min) / (diff + 1e-6)
            
            # If the range is extremely small, the mask should likely be zero or 
            # we should avoid multiplying by trash. 
            # We can zero out norm_mag where diff is very small.
            norm_mag = torch.where(diff > 1e-5, norm_mag, torch.zeros_like(norm_mag))
            
            if self.mask_type == 'hard':
                # Hard binary mask
                mask = (norm_mag > self.threshold).float()
                masks.append(mask)
            else:
                # Soft mask (default)
                masks.append(norm_mag)
            
        return masks

    def forward(self, student_feats, teacher_feats, adapters):
        """
        Args:
            student_feats: list of student feature maps [P3, P4, P5]
            teacher_feats: list of teacher feature maps [P3, P4, P5]
            adapters: ModuleList of 1x1 convs to match student channels to teacher
        """
        loss = 0.0
        
        # Generate masks from Teacher
        masks = self.generate_attention_mask(teacher_feats)
        
        for s_feat, t_feat, adapter, mask in zip(student_feats, teacher_feats, adapters, masks):
            # 1. Adapt student channels to match teacher
            s_adapted = adapter(s_feat)
            
            # 2. Compute MSE weighted by mask
            # mask is [B, 1, H, W], broadcast
            diff = (s_adapted - t_feat)
            weighted_diff = diff * mask
            
            loss += weighted_diff.pow(2).mean()
            
        return loss

class LogitDistillationLoss(nn.Module):
    """
    Vector 2: Logit-Based Response Distillation.
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
        
    def forward(self, student_logits, teacher_logits):
        loss = 0.0
        count = 0
        
        for s_log, t_log in zip(student_logits, teacher_logits):
            # s_log: [B, NC, H, W]
            b, c, h, w = s_log.shape
            s_flat = s_log.permute(0, 2, 3, 1).contiguous().view(-1, c)
            t_flat = t_log.permute(0, 2, 3, 1).contiguous().view(-1, c)
            
            loss += F.kl_div(
                F.log_softmax(s_flat / self.T, dim=-1),
                F.softmax(t_flat / self.T, dim=-1),
                reduction='batchmean'
            ) * (self.T * self.T)
            count += 1
            
        return loss / max(count, 1)

class DFLDistillationLoss(nn.Module):
    """
    Vector 3: Bounding Box Uncertainty Distillation.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, student_dfl, teacher_dfl):
        loss = 0.0
        count = 0
        
        for s_dfl, t_dfl in zip(student_dfl, teacher_dfl):
            # s_dfl: [B, 4*RegMax, H, W]
            b, c, h, w = s_dfl.shape
            # Correct DFL distillation: separate 4 coordinates
            # s_dfl: [B, 4*RegMax, h, w] -> reshape to [B, 4, RegMax, h, w]
            reg_max = c // 4
            s_split = s_dfl.view(b, 4, reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)
            t_split = t_dfl.view(b, 4, reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(-1, 4, reg_max)
            
            loss += F.kl_div(
                F.log_softmax(s_split, dim=-1),
                F.softmax(t_split, dim=-1),
                reduction='batchmean'
            )
            count += 1
            
        return loss / max(count, 1)

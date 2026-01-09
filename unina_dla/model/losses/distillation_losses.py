import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFDistillationLoss(nn.Module):
    """
    Vector 1: Scale-Decoupled Feature (SDF) Distillation.
    Includes spatial attention mask to focus on foreground objects.
    """
    def __init__(self):
        super().__init__()
        
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
            # Normalize to 0-1
            m_min = magnitude.amin(dim=(2,3), keepdim=True)
            m_max = magnitude.amax(dim=(2,3), keepdim=True)
            norm_mag = (magnitude - m_min) / (m_max - m_min + 1e-6)
            
            # Thresholding (e.g., maintain top 40% activations or just soft weight)
            # Using soft mask for better gradient flow
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
            s_flat = s_dfl.permute(0, 2, 3, 1).contiguous().view(-1, c)
            t_flat = t_dfl.permute(0, 2, 3, 1).contiguous().view(-1, c)
            
            loss += F.kl_div(
                F.log_softmax(s_flat, dim=-1),
                F.softmax(t_flat, dim=-1),
                reduction='batchmean'
            )
            count += 1
            
        return loss / max(count, 1)

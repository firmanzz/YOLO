# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Knowledge Distillation for YOLO models.

This module implements knowledge distillation techniques to transfer knowledge
from a large teacher model to a smaller student model.

Usage:
    from ultralytics.utils.distillation import KnowledgeDistiller
    
    distiller = KnowledgeDistiller(teacher_model, student_model, temperature=4.0)
    distillation_loss = distiller(teacher_outputs, student_outputs, targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER


class KnowledgeDistiller:
    """
    Knowledge Distillation for transferring knowledge from teacher to student model.
    
    Implements soft target distillation using temperature scaling and combines
    it with hard target (ground truth) loss.
    
    Attributes:
        teacher_model (nn.Module): Pre-trained teacher model.
        student_model (nn.Module): Student model to train.
        temperature (float): Temperature for softening probability distributions.
        alpha (float): Weight for distillation loss vs. hard target loss.
        
    Example:
        >>> distiller = KnowledgeDistiller(teacher, student, temperature=4.0, alpha=0.7)
        >>> loss = distiller(teacher_out, student_out, targets)
    """
    
    def __init__(self, teacher_model=None, student_model=None, temperature=4.0, alpha=0.7, 
                 distill_features=False):
        """
        Initialize Knowledge Distiller.
        
        Args:
            teacher_model (nn.Module): Teacher model (can be loaded later).
            student_model (nn.Module): Student model.
            temperature (float): Temperature for soft targets (default: 4.0).
            alpha (float): Weight for distillation loss (0.0-1.0, default: 0.7).
            distill_features (bool): Whether to distill intermediate features.
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distill_features = distill_features
        
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            LOGGER.info(f"Knowledge Distiller initialized with T={temperature}, alpha={alpha}")
        else:
            LOGGER.info("Knowledge Distiller initialized (teacher model to be set later)")
    
    def set_teacher_model(self, teacher_model):
        """
        Set or update the teacher model.
        
        Args:
            teacher_model (nn.Module): Teacher model.
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        LOGGER.info("Teacher model set and frozen")
    
    def distillation_loss(self, teacher_logits, student_logits):
        """
        Calculate distillation loss using KL divergence.
        
        Args:
            teacher_logits (torch.Tensor): Teacher model logits.
            student_logits (torch.Tensor): Student model logits.
            
        Returns:
            torch.Tensor: Distillation loss.
        """
        # Soften probabilities with temperature
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature squared (as per Hinton et al.)
        distill_loss = kl_div * (self.temperature ** 2)
        
        return distill_loss
    
    def feature_distillation_loss(self, teacher_features, student_features):
        """
        Calculate feature distillation loss using MSE.
        
        Args:
            teacher_features (list): List of teacher intermediate features.
            student_features (list): List of student intermediate features.
            
        Returns:
            torch.Tensor: Feature distillation loss.
        """
        if not teacher_features or not student_features:
            return torch.tensor(0.0).to(student_features[0].device if student_features else 'cpu')
        
        feature_loss = 0.0
        num_features = min(len(teacher_features), len(student_features))
        
        for t_feat, s_feat in zip(teacher_features[:num_features], student_features[:num_features]):
            # Align dimensions if needed
            if t_feat.shape != s_feat.shape:
                # Simple spatial alignment using adaptive pooling
                if len(t_feat.shape) == 4:  # [B, C, H, W]
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
                    # Channel alignment with 1x1 conv if needed
                    if t_feat.shape[1] != s_feat.shape[1]:
                        continue  # Skip if channel mismatch
            
            # MSE loss between features
            feature_loss += F.mse_loss(s_feat, t_feat.detach())
        
        return feature_loss / num_features if num_features > 0 else feature_loss
    
    def __call__(self, teacher_outputs, student_outputs, hard_loss=None):
        """
        Calculate combined distillation and hard target loss.
        
        Args:
            teacher_outputs (torch.Tensor | tuple): Teacher model outputs.
            student_outputs (torch.Tensor | tuple): Student model outputs.
            hard_loss (torch.Tensor): Hard target loss (ground truth).
            
        Returns:
            torch.Tensor: Combined loss.
        """
        # Handle tuple outputs (logits, features)
        if isinstance(teacher_outputs, tuple):
            teacher_logits, teacher_features = teacher_outputs
        else:
            teacher_logits = teacher_outputs
            teacher_features = []
        
        if isinstance(student_outputs, tuple):
            student_logits, student_features = student_outputs
        else:
            student_logits = student_outputs
            student_features = []
        
        # Calculate distillation loss
        distill_loss = self.distillation_loss(teacher_logits, student_logits)
        
        # Calculate feature distillation loss if enabled
        if self.distill_features and teacher_features and student_features:
            feat_loss = self.feature_distillation_loss(teacher_features, student_features)
            distill_loss = distill_loss + 0.5 * feat_loss
        
        # Combine with hard target loss if provided
        if hard_loss is not None:
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = distill_loss
        
        return total_loss
    
    @torch.no_grad()
    def get_teacher_predictions(self, inputs):
        """
        Get teacher model predictions without gradients.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor | tuple: Teacher predictions.
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not set")
        
        self.teacher_model.eval()
        return self.teacher_model(inputs)


class ResponseBasedDistiller(KnowledgeDistiller):
    """
    Response-based distillation focusing only on final outputs.
    
    This is the standard distillation approach that matches the
    final predictions of teacher and student.
    """
    
    def __init__(self, teacher_model=None, student_model=None, temperature=4.0, alpha=0.7):
        """Initialize Response-based Distiller."""
        super().__init__(teacher_model, student_model, temperature, alpha, distill_features=False)
        LOGGER.info("Response-based Distiller initialized")


class FeatureBasedDistiller(KnowledgeDistiller):
    """
    Feature-based distillation matching intermediate layer features.
    
    This approach transfers knowledge from intermediate representations,
    which can be more effective for similar architectures.
    """
    
    def __init__(self, teacher_model=None, student_model=None, temperature=4.0, alpha=0.7):
        """Initialize Feature-based Distiller."""
        super().__init__(teacher_model, student_model, temperature, alpha, distill_features=True)
        LOGGER.info("Feature-based Distiller initialized")


class YOLODistiller:
    """
    Specialized distillation for YOLO object detection models.
    
    Handles distillation of bounding box predictions, objectness scores,
    and class probabilities in YOLO-style outputs.
    
    Example:
        >>> distiller = YOLODistiller(teacher, student, alpha=0.7)
        >>> loss = distiller.compute_loss(teacher_out, student_out, targets)
    """
    
    def __init__(self, teacher_model=None, student_model=None, alpha=0.7, 
                 bbox_weight=1.0, obj_weight=1.0, cls_weight=1.0):
        """
        Initialize YOLO Distiller.
        
        Args:
            teacher_model (nn.Module): Teacher YOLO model.
            student_model (nn.Module): Student YOLO model.
            alpha (float): Distillation weight.
            bbox_weight (float): Weight for bbox distillation.
            obj_weight (float): Weight for objectness distillation.
            cls_weight (float): Weight for class distillation.
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            LOGGER.info(f"YOLO Distiller initialized with alpha={alpha}")
    
    def distill_bbox_loss(self, teacher_bbox, student_bbox):
        """Distillation loss for bounding box predictions."""
        return F.smooth_l1_loss(student_bbox, teacher_bbox.detach())
    
    def distill_objectness_loss(self, teacher_obj, student_obj):
        """Distillation loss for objectness scores."""
        return F.binary_cross_entropy_with_logits(student_obj, torch.sigmoid(teacher_obj.detach()))
    
    def distill_class_loss(self, teacher_cls, student_cls, temperature=3.0):
        """Distillation loss for class predictions."""
        teacher_probs = F.softmax(teacher_cls / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_cls / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    def compute_loss(self, teacher_outputs, student_outputs, hard_loss=None):
        """
        Compute YOLO distillation loss.
        
        Args:
            teacher_outputs: Teacher model outputs.
            student_outputs: Student model outputs.
            hard_loss: Ground truth loss.
            
        Returns:
            torch.Tensor: Combined loss.
        """
        # This is a simplified version - actual implementation would need to
        # parse YOLO output format and match detection heads
        
        distill_loss = 0.0
        
        # If outputs are tuples/lists (multiple detection heads)
        if isinstance(teacher_outputs, (list, tuple)) and isinstance(student_outputs, (list, tuple)):
            for t_out, s_out in zip(teacher_outputs, student_outputs):
                # Simple MSE distillation on outputs
                distill_loss += F.mse_loss(s_out, t_out.detach())
        else:
            distill_loss = F.mse_loss(student_outputs, teacher_outputs.detach())
        
        # Combine with hard loss
        if hard_loss is not None:
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = distill_loss
        
        return total_loss

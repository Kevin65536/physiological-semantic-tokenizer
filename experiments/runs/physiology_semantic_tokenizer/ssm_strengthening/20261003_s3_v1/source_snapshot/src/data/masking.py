"""
Masking strategies for physiology-semantic training.

Implements three masking patterns:
1. Cross-Modal: Mask modality 1 heavily, keep modality 2 (learns Redundancy)
2. Uni-Modal: Mask modality 1, drop modality 2 entirely (learns Unique)
3. Joint: Random masking on both (learns Synergy)

Reference: docs/pid_mcm_proposal.md (Section 3)
"""
import torch
import numpy as np


class MaskGenerator:
    """
    Generates masks for different training modes.
    """
    def __init__(self, mask_ratio_high=0.8, mask_ratio_low=0.5, seed=None):
        """
        Args:
            mask_ratio_high: Ratio for heavy masking (cross-modal)
            mask_ratio_low: Ratio for light masking (uni/joint)
            seed: Random seed for reproducibility
        """
        self.mask_ratio_high = mask_ratio_high
        self.mask_ratio_low = mask_ratio_low
        self.rng = np.random.RandomState(seed)
        
    def cross_modal_mask(self, batch_size, seq_len):
        """
        Cross-Modal: Mask 80% of X1, keep X2 full.
        
        Returns:
            mask_x1: [Batch, SeqLen] - True for masked positions
            mask_x2: [Batch, SeqLen] - All False (no masking)
        """
        mask_x1 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for i in range(batch_size):
            num_masked = int(seq_len * self.mask_ratio_high)
            masked_indices = self.rng.choice(seq_len, num_masked, replace=False)
            mask_x1[i, masked_indices] = True
            
        mask_x2 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        return mask_x1, mask_x2
    
    def uni_modal_mask(self, batch_size, seq_len, drop_modality=1):
        """
        Uni-Modal: Mask 50% of X1, drop X2 entirely.
        
        Args:
            drop_modality: 0 for X1, 1 for X2
        """
        mask_x1 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_x2 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        if drop_modality == 1:
            # Keep X1 with partial masking, drop X2
            for i in range(batch_size):
                num_masked = int(seq_len * self.mask_ratio_low)
                masked_indices = self.rng.choice(seq_len, num_masked, replace=False)
                mask_x1[i, masked_indices] = True
            mask_x2[:] = True  # Drop all
        else:
            # Keep X2 with partial masking, drop X1
            for i in range(batch_size):
                num_masked = int(seq_len * self.mask_ratio_low)
                masked_indices = self.rng.choice(seq_len, num_masked, replace=False)
                mask_x2[i, masked_indices] = True
            mask_x1[:] = True  # Drop all
            
        return mask_x1, mask_x2
    
    def joint_mask(self, batch_size, seq_len):
        """
        Joint: Random 50% masking on both modalities.
        """
        mask_x1 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_x2 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            num_masked_1 = int(seq_len * self.mask_ratio_low)
            num_masked_2 = int(seq_len * self.mask_ratio_low)
            
            masked_indices_1 = self.rng.choice(seq_len, num_masked_1, replace=False)
            masked_indices_2 = self.rng.choice(seq_len, num_masked_2, replace=False)
            
            mask_x1[i, masked_indices_1] = True
            mask_x2[i, masked_indices_2] = True
            
        return mask_x1, mask_x2


class MixedBatchSampler:
    """
    Creates mixed batches with different masking modes.
    
    Proportions:
    - 25% Cross-Modal
    - 25% Uni-Modal
    - 50% Joint
    """
    def __init__(self, total_batch_size=64, seed=None):
        self.total_batch_size = total_batch_size
        
        # Calculate batch sizes for each mode
        self.batch_cross = int(total_batch_size * 0.25)
        self.batch_uni = int(total_batch_size * 0.25)
        self.batch_joint = total_batch_size - self.batch_cross - self.batch_uni
        
        self.mask_gen = MaskGenerator(seed=seed)
        
    def generate_batch_masks(self, seq_len):
        """
        Generate masks for a mixed batch.
        
        Returns:
            masks: List of (mask_x1, mask_x2, mode) tuples
        """
        masks = []
        
        # Cross-modal
        mask_x1, mask_x2 = self.mask_gen.cross_modal_mask(self.batch_cross, seq_len)
        for i in range(self.batch_cross):
            masks.append((mask_x1[i], mask_x2[i], 'cross'))
        
        # Uni-modal (alternate between dropping X1 and X2)
        half_uni = self.batch_uni // 2
        mask_x1, mask_x2 = self.mask_gen.uni_modal_mask(half_uni, seq_len, drop_modality=1)
        for i in range(half_uni):
            masks.append((mask_x1[i], mask_x2[i], 'uni'))
            
        mask_x1, mask_x2 = self.mask_gen.uni_modal_mask(self.batch_uni - half_uni, seq_len, drop_modality=0)
        for i in range(self.batch_uni - half_uni):
            masks.append((mask_x1[i], mask_x2[i], 'uni'))
        
        # Joint
        mask_x1, mask_x2 = self.mask_gen.joint_mask(self.batch_joint, seq_len)
        for i in range(self.batch_joint):
            masks.append((mask_x1[i], mask_x2[i], 'joint'))
        
        return masks


if __name__ == "__main__":
    # Test
    sampler = MixedBatchSampler(total_batch_size=64)
    masks = sampler.generate_batch_masks(seq_len=100)
    
    print(f"Total samples: {len(masks)}")
    print(f"Cross-modal: {sum(1 for m in masks if m[2] == 'cross')}")
    print(f"Uni-modal: {sum(1 for m in masks if m[2] == 'uni')}")
    print(f"Joint: {sum(1 for m in masks if m[2] == 'joint')}")

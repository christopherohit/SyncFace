"""
InsTaG-inspired Network for SyncTalk: Deformable NeRF with pre-training and adaptation.

This network splits the original SyncTalk architecture into:
1. Motion Fields (UMF + PersonalizedField) - predict spatial deformations
2. Static Field - predict appearance and density at deformed locations

Training happens in two phases:
- Phase 1 (Pre-training): Learn shared UMF across multiple identities
- Phase 2 (Adaptation): Freeze UMF, train MotionAligner + new PersonalizedField + new StaticField
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .instag_modules import (
    UniversalMotionField,
    PersonalizedMotionField,
    StaticField,
    MotionAligner,
    FaceMouthHook,
    create_lip_mask
)
from .network import AudioNet, AudioNet_ave, AudioAttNet, MLP
from .renderer import NeRFRenderer


class InsTaGNetwork(NeRFRenderer):
    """
    InsTaG-inspired talking face network with deformable NeRF.
    
    Architecture:
        - AudioNet: Extracts audio features f_l from speech
        - EyeNet: Extracts eye/expression features f_e (optional)
        - UniversalMotionField (UMF): Shared deformation field (frozen after pre-training)
        - PersonalizedMotionField: Identity-specific refinement
        - MotionAligner: Aligns UMF to new identities (only in adaptation phase)
        - StaticField: Identity-specific appearance and geometry
        - FaceMouthHook: Couples face and mouth region motion (optional)
    
    Forward pass:
        1. Extract audio/eye features: f_l, f_e = AudioNet(audio), EyeNet(eye)
        2. Compute deformations:
           - If adaptation: Delta_x_A, tau_A = MotionAligner(x)
                           delta_univ = UMF(x + Delta_x_A, f_l, f_e) * tau_A
           - Else: delta_univ = UMF(x, f_l, f_e)
           - delta_personal = PersonalizedField(x, f_l, f_e)
           - delta_x = delta_univ + delta_personal
        3. Query static field: sigma, color = StaticField(x + delta_x, d, c)
    """
    
    def __init__(self, opt, phase: str = 'pretrain'):
        """
        Args:
            opt: Options/config object
            phase: 'pretrain' or 'adapt'
        """
        super().__init__(opt)
        
        self.phase = phase
        self.opt = opt
        
        # Audio feature extraction (same as original SyncTalk)
        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_in_dim = 1024
        else:
            self.audio_in_dim = 32
        
        self.emb = self.opt.emb
        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)
        
        self.audio_dim = getattr(opt, 'audio_dim', 32)
        if self.opt.asr_model == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        
        self.att = self.opt.att
        if self.att > 0:
            self.audio_att_net = AudioAttNet(self.audio_dim)
        
        # Eye feature extraction
        self.eye_dim = 0
        if self.opt.exp_eye:
            if self.opt.au45:
                self.eye_dim = 1
            else:
                if self.opt.bs_area == "upper":
                    self.eye_dim = 7
                elif self.opt.bs_area == "single":
                    self.eye_dim = 4
                elif self.opt.bs_area == "eye":
                    self.eye_dim = 2
        
        # Universal Motion Field (shared across identities)
        self.umf = UniversalMotionField(
            audio_dim=self.audio_dim,
            eye_dim=self.eye_dim,
            hidden_dim=64,
            num_layers=3,
            use_hash_encoding=True
        )
        
        # Personalized Motion Field (one per identity during pre-training, one for target in adaptation)
        # During pre-training, we'll create multiple PersonalizedFields (stored in a ModuleDict)
        # During adaptation, we create a single new PersonalizedField
        if phase == 'pretrain':
            # In pre-training, we'll dynamically create PersonalizedFields for each identity
            self.personalized_fields = nn.ModuleDict()
        else:
            # In adaptation, single personalized field for the target identity
            self.personalized_field = PersonalizedMotionField(
                audio_dim=self.audio_dim,
                eye_dim=self.eye_dim,
                hidden_dim=32,
                num_layers=2,
                use_hash_encoding=True
            )
        
        # Static Field (one per identity)
        # Similar structure: ModuleDict for pre-training, single field for adaptation
        if phase == 'pretrain':
            self.static_fields = nn.ModuleDict()
        else:
            self.static_field = StaticField(
                individual_dim=self.individual_dim,
                geo_feat_dim=64,
                hidden_dim=64,
                hidden_dim_color=64,
                num_layers=3,
                num_layers_color=2,
                bound=self.bound
            )
        
        # Motion Aligner (only used in adaptation phase)
        self.motion_aligner = None
        if phase == 'adapt':
            self.motion_aligner = MotionAligner(
                hidden_dim=64,
                num_layers=2,
                scale_init=1.0,
                use_hash_encoding=True
            )
        
        # Face-Mouth Hook (optional, for future enhancement)
        self.use_fm_hook = getattr(opt, 'use_fm_hook', False)
        if self.use_fm_hook:
            self.fm_hook = FaceMouthHook(hook_dim=9)
        else:
            self.fm_hook = None
        
        self.testing = False
    
    def add_identity(self, identity_id: str):
        """
        Add a new identity during pre-training.
        Creates PersonalizedField and StaticField for this identity.
        """
        if self.phase != 'pretrain':
            raise ValueError("Can only add identities during pre-training phase")
        
        # Get device from existing parameters
        device = next(self.parameters()).device
        
        if identity_id not in self.personalized_fields:
            personal_field = PersonalizedMotionField(
                audio_dim=self.audio_dim,
                eye_dim=self.eye_dim,
                hidden_dim=32,
                num_layers=2,
                use_hash_encoding=True
            ).to(device)
            self.personalized_fields[identity_id] = personal_field
        
        if identity_id not in self.static_fields:
            static_field = StaticField(
                individual_dim=self.individual_dim,
                geo_feat_dim=64,
                hidden_dim=64,
                hidden_dim_color=64,
                num_layers=3,
                num_layers_color=2,
                bound=self.bound
            ).to(device)
            self.static_fields[identity_id] = static_field
    
    def encode_audio(self, a):
        """Extract audio features (same as original SyncTalk)."""
        if a is None:
            return None
        
        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous()
        
        enc_a = self.audio_net(a)
        
        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        
        return enc_a
    
    def compute_deformation(self, x: torch.Tensor, f_l: torch.Tensor,
                           f_e: Optional[torch.Tensor] = None,
                           identity_id: Optional[str] = None) -> torch.Tensor:
        """
        Compute total deformation delta_x.
        
        Args:
            x: [N, 3] spatial coordinates
            f_l: [1, audio_dim] audio features
            f_e: [1, eye_dim] optional eye features
            identity_id: Identity ID (only for pre-training phase)
        
        Returns:
            delta_x: [N, 3] total displacement
        """
        # Replicate features to match batch size
        f_l_rep = f_l.repeat(x.shape[0], 1) if f_l is not None else None
        f_e_rep = f_e.repeat(x.shape[0], 1) if f_e is not None else None
        
        # Universal motion field
        if self.phase == 'adapt' and self.motion_aligner is not None:
            # Adaptation: use Motion Aligner
            Delta_x_A, tau_A = self.motion_aligner(x, bound=self.bound)
            x_aligned = x + Delta_x_A
            delta_universal = self.umf(x_aligned, f_l_rep, f_e_rep, bound=self.bound)
            delta_universal = delta_universal * tau_A
        else:
            # Pre-training or no aligner: direct UMF query
            delta_universal = self.umf(x, f_l_rep, f_e_rep, bound=self.bound)
        
        # Personalized motion field
        if self.phase == 'pretrain':
            if identity_id is None:
                raise ValueError("identity_id required in pre-training phase")
            if identity_id not in self.personalized_fields:
                raise ValueError(f"Unknown identity: {identity_id}")
            personal_field = self.personalized_fields[identity_id]
        else:
            personal_field = self.personalized_field
        
        delta_personal = personal_field(x, f_l_rep, f_e_rep, bound=self.bound)
        
        # Total deformation
        delta_x = delta_universal + delta_personal
        
        return delta_x
    
    def forward(self, x: torch.Tensor, d: torch.Tensor, 
                enc_a: torch.Tensor, c: Optional[torch.Tensor] = None,
                e: Optional[torch.Tensor] = None,
                identity_id: Optional[str] = None) -> Tuple:
        """
        Forward pass of InsTaG network.
        
        Args:
            x: [N, 3] spatial coordinates in [-bound, bound]
            d: [N, 3] view directions (normalized)
            enc_a: [1, audio_dim] encoded audio features
            c: [1, individual_dim] optional individual code
            e: [1, eye_dim] optional eye features
            identity_id: Identity ID (for pre-training phase)
        
        Returns:
            sigma: [N,] density
            color: [N, 3] RGB color
            delta_x: [N, 3] deformation (for visualization/analysis)
        """
        # Compute deformation
        delta_x = self.compute_deformation(x, enc_a, e, identity_id)
        
        # Deformed coordinates
        x_deformed = x + delta_x
        
        # Query static field
        if self.phase == 'pretrain':
            if identity_id is None:
                raise ValueError("identity_id required in pre-training phase")
            if identity_id not in self.static_fields:
                raise ValueError(f"Unknown identity: {identity_id}")
            static_field = self.static_fields[identity_id]
        else:
            static_field = self.static_field
        
        sigma, color = static_field(x_deformed, d, c)
        
        return sigma, color, delta_x
    
    def density(self, x: torch.Tensor, enc_a: torch.Tensor,
                e: Optional[torch.Tensor] = None,
                identity_id: Optional[str] = None) -> Dict:
        """
        Query density (used for density grid updates).
        
        Returns dictionary for compatibility with original SyncTalk.
        """
        # Dummy view direction (not used for density)
        d = torch.zeros_like(x)
        d[:, 2] = 1.0  # Forward direction
        
        sigma, _, delta_x = self.forward(x, d, enc_a, None, e, identity_id)
        
        return {
            'sigma': sigma,
            'geo_feat': torch.zeros(x.shape[0], 64, device=x.device),  # Placeholder
            'ambient_aud': torch.zeros(x.shape[0], 1, device=x.device),
            'ambient_eye': torch.zeros(x.shape[0], 1, device=x.device),
        }
    
    def get_params(self, lr, lr_net, wd=0):
        """
        Get parameters for optimizer.
        Different parameter groups for different phases.
        """
        params = []
        
        if self.phase == 'pretrain':
            # Pre-training: train UMF + all PersonalizedFields + all StaticFields
            params.append({'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
            if self.att > 0:
                params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
            if self.emb:
                params.append({'params': self.embedding.parameters(), 'lr': lr})
            
            # UMF (will be frozen later)
            params.append({'params': self.umf.parameters(), 'lr': lr, 'weight_decay': wd})
            
            # All personalized fields and static fields
            params.append({'params': self.personalized_fields.parameters(), 'lr': lr, 'weight_decay': wd})
            params.append({'params': self.static_fields.parameters(), 'lr': lr, 'weight_decay': wd})
            
        else:  # adaptation phase
            # Adaptation: freeze UMF, train MotionAligner + PersonalizedField + StaticField
            # Audio net can be fine-tuned or frozen (frozen here for stability)
            
            # Motion Aligner (key adapter)
            if self.motion_aligner is not None:
                params.append({'params': self.motion_aligner.parameters(), 'lr': lr_net, 'weight_decay': wd})
            
            # New PersonalizedField
            params.append({'params': self.personalized_field.parameters(), 'lr': lr_net, 'weight_decay': wd})
            
            # New StaticField (hash grids + MLPs)
            params.append({'params': self.static_field.parameters(), 'lr': lr, 'weight_decay': wd})
            
            # Individual codes if used
            if self.individual_dim > 0:
                params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        return params
    
    def freeze_umf(self):
        """Freeze UMF parameters (call before adaptation phase)."""
        for param in self.umf.parameters():
            param.requires_grad = False
        print("[INFO] UMF frozen for adaptation phase")
    
    def save_umf_checkpoint(self, path: str):
        """Save only UMF checkpoint (after pre-training)."""
        checkpoint = {
            'umf_state_dict': self.umf.state_dict(),
            'audio_net_state_dict': self.audio_net.state_dict(),
            'audio_dim': self.audio_dim,
            'eye_dim': self.eye_dim,
            'config': {
                'audio_in_dim': self.audio_in_dim,
                'asr_model': self.opt.asr_model,
                'bound': self.bound,
            }
        }
        if self.att > 0:
            checkpoint['audio_att_net_state_dict'] = self.audio_att_net.state_dict()
        if self.emb:
            checkpoint['embedding_state_dict'] = self.embedding.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[INFO] UMF checkpoint saved to {path}")
    
    def load_umf_checkpoint(self, path: str):
        """Load UMF checkpoint (before adaptation phase)."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.umf.load_state_dict(checkpoint['umf_state_dict'])
        self.audio_net.load_state_dict(checkpoint['audio_net_state_dict'])
        
        if self.att > 0 and 'audio_att_net_state_dict' in checkpoint:
            self.audio_att_net.load_state_dict(checkpoint['audio_att_net_state_dict'])
        if self.emb and 'embedding_state_dict' in checkpoint:
            self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        
        print(f"[INFO] UMF checkpoint loaded from {path}")
        self.freeze_umf()


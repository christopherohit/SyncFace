import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder
from .renderer import NeRFRenderer

# Import enhanced audio encoder modules
try:
    from .enhanced_audio_encoder import EnhancedAudioEncoder
    from .audio_encoder_adapter import (
        HybridAudioEncoder, 
        FoundationModelAudioNet, 
        FoundationModelAudioNetAVE
    )
    ENHANCED_ENCODERS_AVAILABLE = True
except ImportError:
    ENHANCED_ENCODERS_AVAILABLE = False
    print("[WARN] Enhanced audio encoders not available. Using original encoders only.")

# Import emotion modules
try:
    from .emotion_module import EmotionRecognitionModule
    from .emotion_integration import EmotionAwareNeRFModule
    EMOTION_MODULES_AVAILABLE = True
except ImportError:
    EMOTION_MODULES_AVAILABLE = False
    print("[WARN] Emotion modules not available. Emotion features disabled.")


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out

# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


# Audio feature extractor
class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )
    def forward(self, x):
        # half_w = int(self.win_size/2)
        # x = x[:, :, 8-half_w:8+half_w]
        # x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).permute(1,0,2).squeeze(0)
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 # Core network parameters
                 emb=False,
                 asr_model='deepspeech',
                 att=2,

                 # Face modeling parameters
                 au45=False,
                 bs_area="upper",
                 exp_eye=False,
                 individual_dim=4,
                 individual_dim_torso=8,

                 # Training parameters
                 torso=False,
                 train_camera=False,

                 # Enhanced audio encoder parameters
                 use_enhanced_encoder=False,
                 enhanced_encoder_type='whisper',
                 use_prosody=True,
                 foundation_model_type='whisper',
                 use_contrastive=False,
                 freeze_audio_backbone=True,

                 # Emotion recognition parameters
                 use_emotion=False,
                 emotion_model='wav2vec2',
                 emotion_checkpoint='',
                 emotion_strength=0.7,

                 # Optional audio_dim override
                 audio_dim=32
                 ):
        """
        Initialize SyncFace NeRF Network with specific parameters instead of passing entire opt object.

        Args:
            emb: Whether to use audio embedding
            asr_model: ASR model type ('deepspeech', 'ave', 'hubert', 'esperanto')
            att: Audio attention mode (0=off, 1=left, 2=bi-direction)

            au45: Use OpenFace AU45 blendshapes
            bs_area: Blendshape area ("upper" or "eye")
            exp_eye: Explicitly control eye movements
            individual_dim: Individual code dimension (0=disable)
            individual_dim_torso: Torso individual code dimension

            torso: Train torso (fix head and train torso)
            train_camera: Optimize camera pose during training

            use_enhanced_encoder: Use enhanced foundation model encoders
            enhanced_encoder_type: Type of enhanced encoder
            use_prosody: Extract prosodic features
            foundation_model_type: Foundation model type for hybrid
            use_contrastive: Use CLIP-like contrastive learning
            freeze_audio_backbone: Freeze foundation model weights

            use_emotion: Enable emotion-aware expressions
            emotion_model: Emotion recognition model type
            emotion_checkpoint: Path to emotion checkpoint
            emotion_strength: Emotion influence strength

            audio_dim: Audio feature dimension (default: 32)
        """
        # Create opt-like object for NeRFRenderer parent class
        # NeRFRenderer expects certain attributes from opt
        class OptWrapper:
            def __init__(self):
                # Scene parameters (will be set by main.py later)
                self.bound = 1.0
                self.min_near = 0.05
                self.density_thresh = 10.0
                self.density_thresh_torso = 0.01
                self.cuda_ray = True

                # Face modeling
                self.au45 = au45
                self.bs_area = bs_area
                self.exp_eye = exp_eye
                self.ind_dim = individual_dim  # NeRFRenderer expects ind_dim
                self.ind_dim_torso = individual_dim_torso

                # Training
                self.torso = torso
                self.train_camera = train_camera
                self.ind_num = 20000  # Default from main.py

                # Test mode (set to False for training)
                self.test_train = False
                self.smooth_lips = False

        opt_wrapper = OptWrapper()

        super().__init__(opt_wrapper)

        # Store all parameters as instance attributes
        self.emb = emb
        self.asr_model = asr_model
        self.att = att
        self.au45 = au45
        self.bs_area = bs_area
        self.exp_eye = exp_eye
        self.individual_dim = individual_dim
        self.individual_dim_torso = individual_dim_torso
        self.torso = torso
        self.train_camera = train_camera

        # Enhanced encoder parameters
        self.use_enhanced_encoder = use_enhanced_encoder
        self.enhanced_encoder_type = enhanced_encoder_type
        self.use_prosody = use_prosody
        self.foundation_model_type = foundation_model_type
        self.use_contrastive = use_contrastive
        self.freeze_audio_backbone = freeze_audio_backbone

        # Emotion parameters
        self.use_emotion = use_emotion
        self.emotion_model = emotion_model
        self.emotion_checkpoint = emotion_checkpoint
        self.emotion_strength = emotion_strength

        # Determine audio input dimension based on ASR model and encoder type
        if 'esperanto' in self.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.asr_model:
            self.audio_in_dim = 1024
        elif self.use_enhanced_encoder and ENHANCED_ENCODERS_AVAILABLE:
            # Enhanced encoders output 512-dim features by default
            self.audio_in_dim = 512
        else:
            self.audio_in_dim = 32

        # Create audio embedding if enabled
        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)

        # Audio feature dimension
        self.audio_dim = audio_dim

        # Initialize audio encoder based on type
        self._init_audio_encoder()

        # Initialize audio attention if enabled
        if self.att > 0:
            self.audio_att_net = AudioAttNet(self.audio_dim)

        # Initialize emotion module if enabled
        self._init_emotion_module()


    def _init_audio_encoder(self):
        """Initialize the appropriate audio encoder."""
        if self.use_enhanced_encoder and ENHANCED_ENCODERS_AVAILABLE:
            print(f"[INFO] Using enhanced audio encoder: {self.enhanced_encoder_type}")

            if self.enhanced_encoder_type == 'hybrid':
                # Hybrid encoder combines LRS2 + foundation models
                self.enhanced_audio_encoder = HybridAudioEncoder(
                    use_lrs2=True,
                    use_foundation=True,
                    foundation_type=self.foundation_model_type,
                    output_dim=self.audio_in_dim
                )
            else:
                # Pure foundation model encoder
                self.enhanced_audio_encoder = EnhancedAudioEncoder(
                    encoder_type=self.enhanced_encoder_type,
                    output_dim=self.audio_in_dim,
                    use_prosody=self.use_prosody,
                    use_contrastive=self.use_contrastive,
                    freeze_backbone=self.freeze_audio_backbone
                )

            # Use enhanced AudioNet for processing
            if self.asr_model == 'ave':
                self.audio_net = FoundationModelAudioNetAVE(self.audio_in_dim, self.audio_dim)
            else:
                self.audio_net = FoundationModelAudioNet(self.audio_in_dim, self.audio_dim)
        else:
            # Use original AudioNet
            if self.asr_model == 'ave':
                self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
            else:
                self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)


    def _init_emotion_module(self):
        """Initialize emotion recognition module if enabled."""
        if not (self.use_emotion and EMOTION_MODULES_AVAILABLE):
            return

        print(f"[INFO] Initializing emotion recognition: {self.emotion_model}")

        self.emotion_module = EmotionAwareNeRFModule(
            emotion_model=self.emotion_model,
            emotion_dim=64,
            emotion_strength=self.emotion_strength,
            use_emotion_conditioning=True
        )

        # Load emotion checkpoint if provided
        if self.emotion_checkpoint and os.path.exists(self.emotion_checkpoint):
            try:
                checkpoint = torch.load(self.emotion_checkpoint, map_location='cpu')
                self.emotion_module.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"[INFO] Loaded emotion checkpoint from {self.emotion_checkpoint}")
            except Exception as e:
                print(f"[WARN] Could not load emotion checkpoint: {e}")

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        if self.opt.au45:
            self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
            self.eye_dim = 1 if self.exp_eye else 0
        else:
            if self.opt.bs_area == "upper":
                self.eye_att_net = MLP(self.in_dim, 7, 64, 2)
                self.eye_dim = 7 if self.exp_eye else 0
            elif self.opt.bs_area == "single":
                self.eye_att_net = MLP(self.in_dim, 4, 64, 2)
                self.eye_dim = 4 if self.exp_eye else 0
            elif self.opt.bs_area == "eye":
                self.eye_att_net = MLP(self.in_dim, 2, 64, 2)
                self.eye_dim = 2 if self.exp_eye else 0
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)

        self.testing = False

        if self.torso:
            # torso deform network
            self.register_parameter('anchor_points', 
                                    nn.Parameter(torch.tensor([[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]])))
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            # self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=1, base_resolution=16, log2_hashmap_size=16, desired_resolution=512)
            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 2, 32, 3)

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
            self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)


    def forward_torso(self, x, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1, -1)
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h)
        
        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1])*(1 + 2*0.001) - 0.001
        color = torch.sigmoid(h[..., 1:])*(1 + 2*0.001) - 0.001

        return alpha, color, dx


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous() # [1/8, 29, 16]

        enc_a = self.audio_net(a) # [8,32]

        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 32]
        
        # Apply emotion conditioning if enabled
        if self.use_emotion and hasattr(self, 'emotion_module'):
            try:
                # Extract emotion from audio features
                # Note: This is a simplified version - you may need to pass raw audio
                emotion_result = self.emotion_module(a.unsqueeze(0))
                emotion_cond = emotion_result['emotion_cond']
                
                # Blend audio features with emotion conditioning
                enc_a = enc_a + self.emotion_strength * emotion_cond[:, :enc_a.shape[-1]]
            except Exception as e:
                # Silently continue if emotion fails
                pass
            
        return enc_a

    
    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc


    def forward(self, x, d, enc_a, c, e=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature
        enc_x = self.encode_x(x, bound=self.bound)

        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
                
        h_color = self.color_net(h)
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        
        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))

        return sigma, color, aud_ch_att, eye_att, uncertainty[..., None]


    def density(self, x, enc_a, e=None, enc_x=None):
        # x: [N, 3], in [-bound, bound]
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att

        if e is not None:
            # e = self.encoder_eye(e)
            # eye_att = torch.sigmoid(self.eye_att_net(enc_x))
            e = e.repeat(enc_x.shape[0], 1)
            eye_att = self.eye_att_net(enc_x)
            e = e * eye_att
            # e = e.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            # {'params': self.encoder_xyz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
        ]
        
        # Add enhanced encoder parameters if using enhanced encoders
        if self.use_enhanced_encoder and ENHANCED_ENCODERS_AVAILABLE:
            if hasattr(self, 'enhanced_audio_encoder'):
                # Lower learning rate for pretrained foundation models
                foundation_lr = lr_net * 0.1 if getattr(self.opt, 'freeze_audio_backbone', True) else lr_net
                params.append({
                    'params': self.enhanced_audio_encoder.parameters(), 
                    'lr': foundation_lr, 
                    'weight_decay': wd * 0.1  # Lower weight decay for foundation models
                })
        
        if self.att > 0:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params

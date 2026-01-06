import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


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


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
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


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
        
        self.cache = None


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

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        
        eye_att = torch.relu(self.eye_att_net(enc_x))
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        
        results = {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }
        self.cache = results
        return results


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 7
        self.move_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim + self.move_dim, self.out_dim, self.hidden_dim, self.num_layers)
        self.scaler_net = MLP(self.in_dim + self.move_dim, 1, 16, 3)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

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


    def forward(self, x, a, move):
        # x: [N, 3], in [-bound, bound]
        # move: [1, D]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # move = torch.as_tensor([[move_max, move_min]]).repeat(enc_x.shape[0], 1).cuda()
        move = move.repeat(enc_x.shape[0], 1)

        h = torch.cat([enc_x, enc_w, move], dim=-1)
        h = self.sigma_net(h)
        
        h_s = torch.cat([enc_x, move], dim=-1)
        h_s = self.scaler_net(h_s)

        d_xyz = h[..., :3] * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        d_rot = h[..., 3:]
        return {
            'd_xyz': d_xyz * torch.sigmoid(h_s) * 2,
            'd_rot': d_rot,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.scaler_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params



# class PersonalizedMotionNetwork(nn.Module):
#     def __init__(self):
#         super(PersonalizedMotionNetwork, self).__init__()
#         self.bound = 0.15
#         self.exp_eye = True

#         # DYNAMIC PART
#         self.num_levels = 12
#         self.level_dim = 1
#         self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
#         self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
#         self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

#         self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        
#         self.num_layers = 3
#         self.hidden_dim = 32

#         self.out_dim = 4
#         self.sigma_net = MLP(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers)

#     @staticmethod
#     @torch.jit.script
#     def split_xyz(x):
#         xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
#         return xy, yz, xz

#     def encode_x(self, xyz, bound):
#         # x: [N, 3], in [-bound, bound]
#         N, M = xyz.shape
#         xy, yz, xz = self.split_xyz(xyz)
#         feat_xy = self.encoder_xy(xy, bound=bound)
#         feat_yz = self.encoder_yz(yz, bound=bound)
#         feat_xz = self.encoder_xz(xz, bound=bound)
        
#         return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


#     def forward(self, x):
#         # x: [N, 3], in [-bound, bound]
#         enc_x = self.encode_x(x, bound=self.bound)
#         h = self.sigma_net(enc_x)

#         d_xyz = h[..., :3] * 1e-2
#         d_scale = h[..., 3:]
#         return {
#             'd_xyz': d_xyz,
#             'd_scale': d_scale,
#         }
        

#     # optimizer utils
#     def get_params(self, lr, lr_net, wd=0):

#         params = [
#             {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
#             {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_yz', 'lr': lr},
#             {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xz', 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'name': 'neural_encoder_net', 'lr': lr_net, 'weight_decay': wd},
#         ]

#         return params



class PersonalizedMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(PersonalizedMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.args = args
        self.bound = 0.15
        self.exp_eye = args.type == 'face'

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3
        self.hidden_dim = 32 if args.type == 'face' else 16
    
        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        if self.exp_eye:
            self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
            self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11 if args.type == 'face' else 7
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        # ============================================================
        # CANONICAL MAPPER: Maps identity space -> canonical space
        # Replaces the old align_net for better disentanglement
        # ============================================================
        self.canonical_encoder, self.canonical_in_dim = get_encoder(
            'hashgrid', 
            input_dim=3, 
            num_levels=12, 
            level_dim=2, 
            base_resolution=16, 
            log2_hashmap_size=17, 
            desired_resolution=256 * self.bound
        )
        # Lightweight MLP that outputs canonical coordinates (3D)
        self.canonical_mlp = MLP(self.canonical_in_dim, 3, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


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

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound] - Identity Space coordinates
        
        # ============================================================
        # STEP 1: Map Identity Space -> Canonical Space
        # This bijective mapping disentangles identity geometry from motion
        # ============================================================
        enc_canon = self.canonical_encoder(x, bound=self.bound)  # Encode identity coords
        x_canon = self.canonical_mlp(enc_canon)  # Map to canonical space [N, 3]
        
        # ============================================================
        # STEP 2: Query Universal Motion Field using Canonical Coordinates
        # All subsequent operations use x_canon instead of x
        # ============================================================
        enc_x = self.encode_x(x_canon, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        h = torch.cat([enc_x, enc_w], dim=-1)

        if self.exp_eye and e is not None:
            eye_att = torch.relu(self.eye_att_net(enc_x))
            enc_e = self.exp_encode_net(e[:-1])
            enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
            enc_e = enc_e * eye_att
            h = torch.cat([h, enc_e], dim=-1)
        elif self.exp_eye and e is None:
            # When exp_eye is enabled but e is not provided (e.g., mouth training)
            # Add zero padding to match expected input size for sigma_net
            h = torch.cat([h, torch.zeros(h.shape[0], self.eye_dim, device=h.device)], dim=-1)
        
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([h, c], dim=-1)

        # ============================================================
        # STEP 3: Predict Deformation in Canonical Space
        # ============================================================
        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        if self.args.type == "face":
            d_opa = h[..., 7:8]
            d_scale = h[..., 8:11]
        else:
            d_opa = d_scale = None
        
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True) if (self.exp_eye and e is not None) else None,
            'x_canon': x_canon,  # Return for regularization during training
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'name': 'neural_audio_net', 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.sigma_net.parameters(), 'name': 'neural_sigma_net', 'lr': lr_net, 'weight_decay': wd},
            # Canonical Mapper parameters (replaces align_net)
            {'params': self.canonical_encoder.parameters(), 'name': 'neural_canonical_encoder', 'lr': lr},
            {'params': self.canonical_mlp.parameters(), 'name': 'neural_canonical_mlp', 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'name': 'neural_audio_att_net', 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'name': 'neural_individual_codes', 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'name': 'neural_aud_ch_att_net', 'lr': lr_net, 'weight_decay': wd})
        
        if self.exp_eye:
            params.append({'params': self.eye_att_net.parameters(), 'name': 'neural_eye_att_net', 'lr': lr_net, 'weight_decay': wd})
            params.append({'params': self.exp_encode_net.parameters(), 'name': 'neural_exp_encode_net', 'lr': lr_net, 'weight_decay': wd})

        return params


class SyncFaceMotionNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        # INPUT DIMS
        self.audio_dim = 32 # Hoặc kích thước feature của HuBERT/AVE
        self.blendshape_dim = 52 # 52 tham số chuẩn ARKit
        self.geo_feat_dim = 64   # Đặc trưng hình học tại điểm 3D (đầu ra của Tri-plane hoặc encoding)
        self.hidden_dim = 64

        # 1. ENCODERS
        # Nhánh Audio
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Nhánh Blendshapes (Face)
        self.face_encoder = nn.Sequential(
            nn.Linear(self.blendshape_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 2. ATTENTION NETWORKS (Spatial Mask Predictors)
        # Input: Tọa độ điểm 3D (xyz) hoặc Geometric Feature
        # Output: 1 giá trị (Mask) từ 0 đến 1
        self.audio_att_net = nn.Sequential(
            nn.Linear(3, 32), # Input là xyz
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Ép về [0, 1] -> Mask
        )

        self.face_att_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 3. DEFORMATION DECODER
        # Input: Feature đã trộn
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 3, 64), # +3 cho xyz gốc
            nn.ReLU(),
            nn.Linear(64, 3 + 4 + 1) # Output: d_xyz (3), d_rot (4), d_scale (1)
        )

    def forward(self, xyz, audio_emb, blendshapes):
        # xyz: [N, 3] - Tọa độ các điểm Gaussian
        # audio_emb: [1, 32] - Feature âm thanh
        # blendshapes: [1, 52] - Tham số biểu cảm

        N = xyz.shape[0]

        # A. Encode Tín hiệu
        feat_aud = self.audio_encoder(audio_emb) # [1, 64]
        feat_face = self.face_encoder(blendshapes) # [1, 64]

        # B. Tính Mask (Attention) tại từng điểm
        # Mỗi điểm 3D sẽ có mask riêng
        mask_aud = self.audio_att_net(xyz) # [N, 1]
        mask_face = self.face_att_net(xyz) # [N, 1]

        # C. Áp dụng Attention (Broadcasting)
        # Nhân feature với mask tương ứng
        # feat_aud (toàn cục) * mask_aud (cục bộ)
        weighted_aud = feat_aud.repeat(N, 1) * mask_aud # [N, 64]
        weighted_face = feat_face.repeat(N, 1) * mask_face # [N, 64]

        # D. Ghép nối (Fusion)
        # Kết hợp thông tin hình học + âm thanh đã lọc + biểu cảm đã lọc
        fused_feat = torch.cat([xyz, weighted_aud, weighted_face], dim=-1) # [N, 3+64+64]

        # E. Dự đoán Biến dạng (Deformation)
        outputs = self.decoder(fused_feat)

        d_xyz = outputs[:, :3]
        d_rot = outputs[:, 3:7]
        d_scale = outputs[:, 7:]

        # F. Kẹp giá trị (Safety Clamp - Chống NaN)
        d_xyz = torch.clamp(d_xyz, -0.1, 0.1)

        return {
            'd_xyz': d_xyz,
            'd_rotation': d_rot,
            'd_scaling': d_scale,
            'mask_aud': mask_aud, # Trả về để visualize debug
            'mask_face': mask_face
        }

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.audio_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.face_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.audio_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.face_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.decoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        return params

"""
SyncGaussian Enhanced Preprocessing Pipeline
============================================
Comprehensive preprocessing for hybrid 3DGS-NeRF talking head synthesis.

Outputs:
- audio_feats.npy: [T, F] audio features (HuBERT/AVE based)
- blendshapes.npy: [T, 52] ARKit blendshape coefficients  
- lips_rect.json: per-frame mouth bounding boxes
- mouth_mask/*.png: binary mouth masks
- transforms_train.json: camera poses from bundle adjustment
- gt_imgs/: inpainted ground truth images
- torso_imgs/: torso inpainted images

Usage:
    python process_sync.py <video_path> --task -1 --asr hubert
"""

import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import face_alignment
except ImportError:
    print("[WARNING] face_alignment not installed. Run: pip install face-alignment")

from face_tracking.util import euler2rot


def extract_audio(path, out_path, sample_rate=16000):
    """Extract audio from video file."""
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -y -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path, out_dir, mode='hubert'):
    """
    Extract audio features using HuBERT or AVE encoder.
    
    Args:
        path: Path to audio file
        out_dir: Output directory
        mode: 'hubert', 'ave', or 'deepspeech'
    """
    print(f'[INFO] ===== extract audio features for {path} using {mode} =====')
    
    if mode == 'ave':
        # AVE is integrated into training, save placeholder
        print(f'AVE features computed during training')
        # Save empty placeholder to indicate AVE mode
        np.save(os.path.join(out_dir, 'audio_feats_ave.npy'), np.array([]))
        
    elif mode == 'hubert':
        # Use HuBERT for feature extraction
        cmd = f'python data_utils/hubert.py --wav {path}'
        os.system(cmd)
        
        # Rename output to standardized name
        hu_path = path.replace('.wav', '_hu.npy')
        if os.path.exists(hu_path):
            feats = np.load(hu_path)
            np.save(os.path.join(out_dir, 'audio_feats.npy'), feats.astype(np.float32))
            print(f'[INFO] Saved HuBERT features: shape {feats.shape}')
            
    elif mode == 'deepspeech':
        cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
        os.system(cmd)
        
    print(f'[INFO] ===== extracted audio features =====')


def extract_images(path, out_path, fps=25):
    """Extract video frames."""
    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -y -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir):
    """Run face parsing to get semantic masks."""
    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_mouth_masks(parsing_dir, mouth_mask_dir):
    """
    Extract binary mouth masks from semantic parsing results.
    
    Saves per-frame binary masks (0/255) for mouth region.
    """
    print(f'[INFO] ===== extract mouth masks from {parsing_dir} =====')
    
    os.makedirs(mouth_mask_dir, exist_ok=True)
    
    parsing_files = sorted(glob.glob(os.path.join(parsing_dir, '*.png')), 
                          key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    for parsing_path in tqdm.tqdm(parsing_files):
        frame_idx = os.path.basename(parsing_path).split('.')[0]
        
        # Load parsing result
        parsing = cv2.imread(parsing_path)
        
        # Mouth region is typically encoded as specific color
        # Gray (100, 100, 100) indicates mouth interior
        mouth_mask = ((parsing[:, :, 0] == 100) & 
                     (parsing[:, :, 1] == 100) & 
                     (parsing[:, :, 2] == 100))
        
        # Also include teeth if present
        teeth_mask_path = parsing_path.replace('parsing', 'teeth_mask').replace('.png', '.npy')
        if os.path.exists(teeth_mask_path):
            teeth_mask = np.load(teeth_mask_path)
            mouth_mask = mouth_mask | teeth_mask
        
        # Save as binary mask (0 or 255)
        mask_out = (mouth_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mouth_mask_dir, f'{frame_idx}.png'), mask_out)
    
    print(f'[INFO] ===== extracted {len(parsing_files)} mouth masks =====')


def extract_landmarks(ori_imgs_dir):
    """Extract 68-point face landmarks."""
    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')
    
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    for image_path in tqdm.tqdm(image_paths):
        input_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input_img)
        if preds and len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:, :2]
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
    
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')


def extract_lips_rect(ori_imgs_dir, out_path):
    """
    Extract per-frame mouth bounding boxes from landmarks.
    
    Saves lips_rect.json with format:
    [{"frame": idx, "rect": [x_min, y_min, x_max, y_max]}, ...]
    """
    print(f'[INFO] ===== extract lips bounding boxes =====')
    
    lms_files = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.lms')),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    lips_rect_list = []
    
    for lms_path in tqdm.tqdm(lms_files):
        frame_idx = int(os.path.basename(lms_path).split('.')[0])
        
        try:
            lms = np.loadtxt(lms_path)  # [68, 2]
            
            # Lip landmarks are indices 48-68
            lips = lms[48:68]  # outer and inner lip
            
            x_min = int(lips[:, 0].min())
            x_max = int(lips[:, 0].max())
            y_min = int(lips[:, 1].min())
            y_max = int(lips[:, 1].max())
            
            # Add margin (10% padding)
            w = x_max - x_min
            h = y_max - y_min
            margin = int(max(w, h) * 0.15)
            
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = x_max + margin
            y_max = y_max + margin
            
            lips_rect_list.append({
                "frame": frame_idx,
                "rect": [x_min, y_min, x_max, y_max]
            })
            
        except Exception as e:
            print(f"[WARNING] Failed to process {lms_path}: {e}")
            continue
    
    # Save to JSON
    with open(out_path, 'w') as f:
        json.dump(lips_rect_list, f, indent=2)
    
    print(f'[INFO] ===== extracted {len(lips_rect_list)} lips rectangles =====')


def extract_background(base_dir, ori_imgs_dir):
    """Extract background image using nearest neighbor inpainting."""
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')
    
    from sklearn.neighbors import NearestNeighbors
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    image_paths = image_paths[::20]  # Use every 20th frame
    
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = tmp_image.shape[:2]
    
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)
    
    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)
    
    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]
    
    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
    
    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)
    
    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    
    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)
    print(f'[INFO] ===== extracted background image =====')


def extract_torso_and_gt(base_dir, ori_imgs_dir):
    """Extract torso and ground truth images with inpainting."""
    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')
    
    from scipy.ndimage import binary_dilation
    
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    for image_path in tqdm.tqdm(image_paths):
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        mask_img = np.zeros_like(seg)
        
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)
        
        mask_img[head_part, :] = 255
        cv2.imwrite(image_path.replace('ori_imgs', 'face_mask').replace('.jpg', '.png'), mask_img)
        
        # Get GT image with background inpainted
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)
        
        # Get torso image with head region inpainted
        torso_image = gt_image.copy()
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8)
        
        # Vertical inpainting for torso
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1)
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid]
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)]
        
        inpaint_torso_mask = None
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            top_torso_colors = gt_image[tuple(top_torso_coords.T)]
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0)
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0)
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3)
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors
            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        
        # Neck inpainting
        push_down = 4
        L = 48 + push_down + 1
        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)
        
        neck_coords = np.stack(np.nonzero(neck_part), axis=-1)
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)]
        
        top_neck_coords = top_neck_coords[mask]
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        top_neck_colors = gt_image[tuple(top_neck_coords.T)]
        
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)
        
        neck_avg_color = np.mean(gt_image[neck_part], axis=0)
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)
        alpha_values = np.linspace(1, 0, L).reshape(L, 1, 1)
        inpaint_neck_colors = inpaint_neck_colors * alpha_values + neck_avg_color * (1 - alpha_values)
        inpaint_neck_colors = inpaint_neck_colors.reshape(-1, 3)
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors
        
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True
        
        blur_img = cv2.GaussianBlur(torso_image.copy(), (5, 5), cv2.BORDER_DEFAULT)
        torso_image[inpaint_mask] = blur_img[inpaint_mask]
        
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0
        
        cv2.imwrite(image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'), 
                   np.concatenate([torso_image, torso_alpha], axis=-1))
    
    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir):
    """Run face tracking to get initial poses."""
    print(f'[INFO] ===== perform face tracking =====')
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = tmp_image.shape[:2]
    
    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'
    os.system(cmd)
    
    print(f'[INFO] ===== finished face tracking =====')


def extract_flow(base_dir, ori_imgs_dir, mask_dir, flow_dir):
    """Extract optical flow and run bundle adjustment for pose refinement."""
    print(f'[INFO] ===== extract flow and run bundle adjustment =====')
    
    torch.cuda.empty_cache()
    ref_id = 2
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = tmp_image.shape[:2]
    
    valid_img_ids = []
    for i in range(100000):
        if os.path.isfile(os.path.join(ori_imgs_dir, '{:d}.lms'.format(i))):
            valid_img_ids.append(i)
    valid_img_num = len(valid_img_ids)
    
    # Create flow list file
    with open(os.path.join(base_dir, 'flow_list.txt'), 'w') as file:
        for i in range(0, valid_img_num):
            file.write(f"{base_dir}/ori_imgs/{ref_id:d}.jpg "
                      f"{base_dir}/face_mask/{ref_id:d}.png "
                      f"{base_dir}/ori_imgs/{i:d}.jpg "
                      f"{base_dir}/face_mask/{i:d}.png\n")
    
    # Run optical flow extraction
    ext_flow_cmd = (f'python data_utils/UNFaceFlow/test_flow.py '
                   f'--datapath={base_dir}/flow_list.txt '
                   f'--savepath={base_dir}/flow_result '
                   f'--width={w} --height={h}')
    os.system(ext_flow_cmd)
    
    # Process flow for bundle adjustment
    face_img = cv2.imread(os.path.join(ori_imgs_dir, f'{ref_id:d}.jpg'))
    face_img_mask = cv2.imread(os.path.join(mask_dir, f'{ref_id:d}.png'))
    
    rigid_mask = face_img_mask[..., 0] > 250
    rigid_num = np.sum(rigid_mask)
    flow_frame_num = min(2500, valid_img_num)
    
    rigid_flow = np.zeros((flow_frame_num, 2, rigid_num), np.float32)
    for i in range(flow_frame_num):
        flow = np.load(os.path.join(flow_dir, f'{ref_id:d}_{valid_img_ids[i]:d}.npy'))
        rigid_flow[i] = flow[:, rigid_mask]
    
    rigid_flow = rigid_flow.transpose((2, 1, 0))
    rigid_flow = torch.as_tensor(rigid_flow).cuda()
    
    lap_kernel = torch.Tensor((-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().cuda()
    flow_lap = F.conv1d(rigid_flow.reshape(-1, 1, rigid_flow.shape[-1]), lap_kernel)
    flow_lap = flow_lap.view(rigid_flow.shape[0], 2, -1)
    flow_lap = torch.norm(flow_lap, dim=1)
    valid_frame = torch.mean(flow_lap, dim=0) < (torch.mean(flow_lap) * 3)
    flow_lap = flow_lap[:, valid_frame]
    
    rigid_flow_mean = torch.mean(flow_lap, dim=1)
    rigid_flow_show = (rigid_flow_mean - torch.min(rigid_flow_mean)) / \
                      (torch.max(rigid_flow_mean) - torch.min(rigid_flow_mean)) * 255
    rigid_flow_show = rigid_flow_show.byte().cpu().numpy()
    
    rigid_flow_img = np.zeros((h, w, 1), dtype=np.uint8)
    rigid_flow_img[...] = 255
    rigid_flow_img[rigid_mask, 0] = rigid_flow_show
    cv2.imwrite(os.path.join(base_dir, 'rigid_flow.jpg'), rigid_flow_img)
    
    # Select keypoints for bundle adjustment
    win_size, d_size = 5, 5
    sel_xys = np.zeros((h, w), dtype=np.int32)
    xys = []
    
    for y in range(0, h - win_size, win_size):
        for x in range(0, w - win_size, win_size):
            min_v = int(40)
            id_x, id_y = -1, -1
            for dy in range(0, win_size):
                for dx in range(0, win_size):
                    if rigid_flow_img[y + dy, x + dx, 0] < min_v:
                        min_v = rigid_flow_img[y + dy, x + dx, 0]
                        id_x = x + dx
                        id_y = y + dy
            if id_x >= 0:
                if np.sum(sel_xys[id_y - d_size:id_y + d_size + 1, id_x - d_size:id_x + d_size + 1]) == 0:
                    cv2.circle(face_img, (id_x, id_y), 1, (255, 0, 0))
                    xys.append(np.array((id_x, id_y), np.int32))
                    sel_xys[id_y, id_x] = 1
    
    cv2.imwrite(os.path.join(base_dir, 'keypts.jpg'), face_img)
    np.savetxt(os.path.join(base_dir, 'keypoints.txt'), xys, '%d')
    
    # Track keypoints
    key_xys = np.loadtxt(os.path.join(base_dir, 'keypoints.txt'), np.int32)
    track_xys = np.zeros((valid_img_num, key_xys.shape[0], 2), dtype=np.float32)
    track_dir = os.path.join(base_dir, 'flow_result')
    track_paths = sorted(glob.glob(os.path.join(track_dir, '*.npy')), 
                        key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0]))
    
    for i, path in enumerate(track_paths):
        flow = np.load(path)
        for j in range(key_xys.shape[0]):
            x, y = key_xys[j, 0], key_xys[j, 1]
            track_xys[i, j, 0] = x + flow[0, y, x]
            track_xys[i, j, 1] = y + flow[1, y, x]
    
    np.save(os.path.join(base_dir, 'track_xys.npy'), track_xys)
    
    # Run bundle adjustment
    pose_opt_cmd = f'python data_utils/face_tracking/bundle_adjustment.py --path={base_dir} --img_h={h} --img_w={w}'
    os.system(pose_opt_cmd)
    
    print(f'[INFO] ===== finished flow extraction and bundle adjustment =====')


def extract_blendshape(base_dir):
    """
    Extract ARKit-compatible 52-dimension blendshape coefficients.
    
    Output: blendshapes.npy with shape [T, 52]
    """
    print(f'[INFO] ===== extract blendshapes =====')
    
    blendshape_cmd = f'python data_utils/blendshape_capture/main.py --path={base_dir}'
    os.system(blendshape_cmd)
    
    # Verify output
    bs_path = os.path.join(base_dir, 'blendshapes.npy')
    if os.path.exists(bs_path):
        bs = np.load(bs_path)
        print(f'[INFO] Blendshapes shape: {bs.shape}')
    else:
        print(f'[WARNING] blendshapes.npy not found. Creating placeholder.')
        # Create placeholder with 52 dims
        ori_imgs = glob.glob(os.path.join(base_dir, 'ori_imgs', '*.jpg'))
        num_frames = len(ori_imgs)
        bs = np.zeros((num_frames, 52), dtype=np.float32)
        np.save(bs_path, bs)
    
    print(f'[INFO] ===== extracted blendshapes =====')


def save_transforms(base_dir, ori_imgs_dir):
    """
    Save camera transforms from bundle adjustment results.
    
    Output: transforms_train.json and transforms_val.json
    """
    print(f'[INFO] ===== save transforms =====')
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = tmp_image.shape[:2]
    
    params_dict = torch.load(os.path.join(base_dir, 'bundle_adjustment.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans']
    valid_num = euler_angle.shape[0]
    
    train_val_split = int(valid_num * 10 / 11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)
    
    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
    
    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    
    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w / 2.0)
        transform_dict['cy'] = float(h / 2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]
        
        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i
            
            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]
            
            frame_dict['transform_matrix'] = pose.numpy().tolist()
            transform_dict['frames'].append(frame_dict)
        
        with open(os.path.join(base_dir, f'transforms_{save_id}.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))
    
    print(f'[INFO] ===== saved transforms =====')


def create_data_summary(base_dir):
    """
    Create a summary of all preprocessed data for verification.
    """
    print(f'[INFO] ===== creating data summary =====')
    
    summary = {
        'base_dir': base_dir,
        'files': {}
    }
    
    # Check required files
    required_files = [
        ('audio_feats.npy', 'Audio features'),
        ('blendshapes.npy', 'Blendshape coefficients'),
        ('lips_rect.json', 'Lip bounding boxes'),
        ('transforms_train.json', 'Training camera poses'),
        ('transforms_val.json', 'Validation camera poses'),
        ('bc.jpg', 'Background image'),
        ('bundle_adjustment.pt', 'Bundle adjustment results'),
    ]
    
    for fname, desc in required_files:
        fpath = os.path.join(base_dir, fname)
        exists = os.path.exists(fpath)
        summary['files'][fname] = {
            'exists': exists,
            'description': desc
        }
        
        if exists and fname.endswith('.npy'):
            arr = np.load(fpath)
            summary['files'][fname]['shape'] = arr.shape
    
    # Check directories
    required_dirs = [
        ('ori_imgs', 'Original images'),
        ('gt_imgs', 'Ground truth images'),
        ('torso_imgs', 'Torso images'),
        ('parsing', 'Semantic parsing'),
        ('face_mask', 'Face masks'),
        ('mouth_mask', 'Mouth masks'),
    ]
    
    for dname, desc in required_dirs:
        dpath = os.path.join(base_dir, dname)
        exists = os.path.isdir(dpath)
        num_files = len(glob.glob(os.path.join(dpath, '*'))) if exists else 0
        summary['files'][dname] = {
            'exists': exists,
            'is_directory': True,
            'num_files': num_files,
            'description': desc
        }
    
    # Save summary
    summary_path = os.path.join(base_dir, 'data_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA PREPROCESSING SUMMARY")
    print("="*60)
    for fname, info in summary['files'].items():
        status = "✓" if info['exists'] else "✗"
        shape_info = f" shape={info.get('shape', 'N/A')}" if 'shape' in info else ""
        files_info = f" ({info['num_files']} files)" if 'num_files' in info else ""
        print(f"  [{status}] {fname}: {info['description']}{shape_info}{files_info}")
    print("="*60 + "\n")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SyncGaussian Preprocessing Pipeline")
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, 
                       help="Task to run (-1=all, 1=audio, 2=images, 3=semantics, "
                            "4=background, 5=torso/gt, 6=landmarks, 7=tracking, "
                            "8=flow/BA, 9=blendshapes, 10=transforms, 11=lips_rect, "
                            "12=mouth_mask, 13=summary)")
    parser.add_argument('--asr', type=str, default='hubert',
                       choices=['hubert', 'ave', 'deepspeech'],
                       help="Audio feature extractor")
    
    opt = parser.parse_args()
    
    base_dir = os.path.dirname(opt.path)
    
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    mask_imgs_dir = os.path.join(base_dir, 'face_mask')
    mouth_mask_dir = os.path.join(base_dir, 'mouth_mask')
    flow_dir = os.path.join(base_dir, 'flow_result')
    lips_rect_path = os.path.join(base_dir, 'lips_rect.json')
    
    # Create directories
    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)
    os.makedirs(mask_imgs_dir, exist_ok=True)
    os.makedirs(mouth_mask_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    
    # Task 1: Extract audio
    if opt.task == -1 or opt.task == 1:
        extract_audio(opt.path, wav_path)
        extract_audio_features(wav_path, base_dir, mode=opt.asr)
    
    # Task 2: Extract images
    if opt.task == -1 or opt.task == 2:
        extract_images(opt.path, ori_imgs_dir)
    
    # Task 3: Face parsing
    if opt.task == -1 or opt.task == 3:
        extract_semantics(ori_imgs_dir, parsing_dir)
    
    # Task 4: Extract background
    if opt.task == -1 or opt.task == 4:
        extract_background(base_dir, ori_imgs_dir)
    
    # Task 5: Extract torso and GT images
    if opt.task == -1 or opt.task == 5:
        extract_torso_and_gt(base_dir, ori_imgs_dir)
    
    # Task 6: Extract landmarks
    if opt.task == -1 or opt.task == 6:
        extract_landmarks(ori_imgs_dir)
    
    # Task 7: Face tracking
    if opt.task == -1 or opt.task == 7:
        face_tracking(ori_imgs_dir)
    
    # Task 8: Extract flow and bundle adjustment
    if opt.task == -1 or opt.task == 8:
        extract_flow(base_dir, ori_imgs_dir, mask_imgs_dir, flow_dir)
    
    # Task 9: Extract blendshapes
    if opt.task == -1 or opt.task == 9:
        extract_blendshape(base_dir)
    
    # Task 10: Save transforms
    if opt.task == -1 or opt.task == 10:
        save_transforms(base_dir, ori_imgs_dir)
    
    # Task 11: Extract lips rectangles
    if opt.task == -1 or opt.task == 11:
        extract_lips_rect(ori_imgs_dir, lips_rect_path)
    
    # Task 12: Extract mouth masks
    if opt.task == -1 or opt.task == 12:
        extract_mouth_masks(parsing_dir, mouth_mask_dir)
    
    # Task 13: Create data summary
    if opt.task == -1 or opt.task == 13:
        create_data_summary(base_dir)


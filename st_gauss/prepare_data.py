"""
ST-Gauss Data Preparation Pipeline

Uses SyncTalk's preprocessing instead of TalkingGaussian's standard pipeline.
This provides:
- Head-Sync Stabilizer (bundle adjustment) for stable poses
- 52 ARKit blendshape parameters (vs simple Action Units)
- AVE audio features (trained on LRS2 for visual sync)

Steps:
1. Extract audio and frames
2. Extract semantics (face parsing)
3. Extract landmarks
4. Face tracking with bundle adjustment
5. Blendshape capture
6. Generate transforms.json
"""

import os
import sys
import glob
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000):
    """Extract audio from video file."""
    print(f'[INFO] Extracting audio from {video_path}')
    cmd = f'ffmpeg -i {video_path} -f wav -ar {sample_rate} -y {output_path}'
    os.system(cmd)
    print(f'[INFO] Audio saved to {output_path}')


def extract_frames(video_path: str, output_dir: str, fps: int = 25):
    """Extract frames from video at specified FPS."""
    print(f'[INFO] Extracting frames at {fps} FPS')
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'ffmpeg -i {video_path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(output_dir, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] Frames saved to {output_dir}')


def extract_semantics(ori_imgs_dir: str, parsing_dir: str):
    """Extract face parsing/segmentation masks."""
    print(f'[INFO] Extracting semantics from {ori_imgs_dir}')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] Parsing masks saved to {parsing_dir}')


def extract_landmarks(ori_imgs_dir: str):
    """Extract 2D face landmarks using face_alignment."""
    print(f'[INFO] Extracting face landmarks')
    try:
        import face_alignment
        try:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        except:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        
        image_paths = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.jpg')))
        
        for image_path in tqdm(image_paths, desc="Extracting landmarks"):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            preds = fa.get_landmarks(img_rgb)
            if preds is not None and len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:, :2]
                lms_path = image_path.replace('.jpg', '.lms')
                np.savetxt(lms_path, lands, '%f')
        
        del fa
        print(f'[INFO] Landmarks extraction complete')
    except ImportError:
        print('[WARNING] face_alignment not installed, skipping landmark extraction')


def extract_background(base_dir: str, ori_imgs_dir: str):
    """Extract static background image."""
    print(f'[INFO] Extracting background image')
    
    from sklearn.neighbors import NearestNeighbors
    
    image_paths = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.jpg')))
    # Sample every 20th frame
    image_paths = image_paths[::20]
    
    tmp_image = cv2.imread(image_paths[0])
    h, w = tmp_image.shape[:2]
    
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    
    for image_path in tqdm(image_paths, desc="Computing background"):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        if parse_img is None:
            continue
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        
        if fg_xys.shape[0] == 0:
            continue
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)
    
    if len(distss) == 0:
        print('[WARNING] No valid parsing masks found')
        return
    
    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)
    
    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]
    
    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths[:len(distss)]:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
    
    bc_img = np.zeros((h * w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)
    
    # Fill in holes
    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    
    if fg_xys.shape[0] > 0 and bg_xys.shape[0] > 0:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        _, indices = nbrs.kneighbors(bg_xys)
        bg_fg_xys = fg_xys[indices[:, 0]]
        bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    
    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)
    print(f'[INFO] Background saved to {os.path.join(base_dir, "bc.jpg")}')


def run_face_tracking(ori_imgs_dir: str, base_dir: str):
    """Run face tracking with 3DMM fitting."""
    print(f'[INFO] Running face tracking')
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0])
    h, w = tmp_image.shape[:2]
    
    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'
    os.system(cmd)
    print(f'[INFO] Face tracking complete')


def run_bundle_adjustment(base_dir: str, ori_imgs_dir: str, mask_dir: str, flow_dir: str):
    """
    Run SyncTalk's bundle adjustment for stable camera poses.
    This is the Head-Sync Stabilizer from SyncTalk.
    """
    print(f'[INFO] Running bundle adjustment (Head-Sync Stabilizer)')
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0])
    h, w = tmp_image.shape[:2]
    
    # First extract optical flow
    valid_img_ids = []
    for i in range(100000):
        if os.path.isfile(os.path.join(ori_imgs_dir, f'{i}.lms')):
            valid_img_ids.append(i)
    
    valid_img_num = len(valid_img_ids)
    ref_id = 2
    
    # Create flow list
    with open(os.path.join(base_dir, 'flow_list.txt'), 'w') as f:
        for i in range(valid_img_num):
            f.write(f'{base_dir}/ori_imgs/{ref_id}.jpg '
                   f'{base_dir}/face_mask/{ref_id}.png '
                   f'{base_dir}/ori_imgs/{i}.jpg '
                   f'{base_dir}/face_mask/{i}.png\n')
    
    # Extract flow
    print('[INFO] Extracting optical flow...')
    flow_cmd = (f'python data_utils/UNFaceFlow/test_flow.py '
                f'--datapath={base_dir}/flow_list.txt '
                f'--savepath={flow_dir} '
                f'--width={w} --height={h}')
    os.system(flow_cmd)
    
    # Generate keypoints and track_xys from flow (required for bundle adjustment)
    print('[INFO] Generating keypoints and track coordinates...')
    face_img = cv2.imread(os.path.join(ori_imgs_dir, f'{ref_id}.jpg'))
    face_img_mask = cv2.imread(os.path.join(mask_dir, f'{ref_id}.png'))
    
    rigid_mask = face_img_mask[..., 0] > 250
    rigid_num = np.sum(rigid_mask)
    flow_frame_num = min(2500, valid_img_num)
    rigid_flow = np.zeros((flow_frame_num, 2, rigid_num), np.float32)
    
    for i in range(flow_frame_num):
        flow_path = os.path.join(flow_dir, f'{ref_id}_{valid_img_ids[i]}.npy')
        if os.path.exists(flow_path):
            flow = np.load(flow_path)
            rigid_flow[i] = flow[:, rigid_mask]
    
    rigid_flow = rigid_flow.transpose((2, 1, 0))
    rigid_flow_tensor = torch.as_tensor(rigid_flow).cuda()
    lap_kernel = torch.Tensor((-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().cuda()
    flow_lap = F.conv1d(rigid_flow_tensor.reshape(-1, 1, rigid_flow_tensor.shape[-1]), lap_kernel)
    flow_lap = flow_lap.view(rigid_flow_tensor.shape[0], 2, -1)
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
    
    # Extract keypoints
    win_size, d_size = 5, 5
    sel_xys = np.zeros((h, w), dtype=np.int32)
    xys = []
    for y in range(0, h - win_size, win_size):
        for x in range(0, w - win_size, win_size):
            min_v = int(40)
            id_x = -1
            id_y = -1
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
    
    # Generate track_xys
    key_xys = np.loadtxt(os.path.join(base_dir, 'keypoints.txt'), np.int32)
    track_xys = np.zeros((valid_img_num, key_xys.shape[0], 2), dtype=np.float32)
    track_paths = sorted(
        glob.glob(os.path.join(flow_dir, f'{ref_id}_*.npy')),
        key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0].split('_')[-1])
    )
    
    for i, path in enumerate(track_paths):
        flow = np.load(path)
        for j in range(key_xys.shape[0]):
            x = key_xys[j, 0]
            y = key_xys[j, 1]
            track_xys[i, j, 0] = x + flow[0, y, x]
            track_xys[i, j, 1] = y + flow[1, y, x]
    
    np.save(os.path.join(base_dir, 'track_xys.npy'), track_xys)
    print(f'[INFO] Generated track_xys.npy with shape {track_xys.shape}')
    
    # Run bundle adjustment
    print('[INFO] Running pose optimization...')
    pose_cmd = f'python data_utils/face_tracking/bundle_adjustment.py --path={base_dir} --img_h={h} --img_w={w}'
    os.system(pose_cmd)
    
    print(f'[INFO] Bundle adjustment complete')


def run_blendshape_capture(base_dir: str):
    """
    Run SyncTalk's blendshape capture for 52 ARKit parameters.
    This replaces simple Action Units with granular expression control.
    """
    print(f'[INFO] Running blendshape capture (52 ARKit parameters)')
    
    cmd = f'python data_utils/blendshape_capture/main.py --path={base_dir}'
    os.system(cmd)
    
    print(f'[INFO] Blendshape capture complete')


def extract_torso_and_masks(base_dir: str, ori_imgs_dir: str):
    """Extract torso images and detailed masks."""
    print(f'[INFO] Extracting torso and masks')
    
    from scipy.ndimage import binary_dilation
    
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'))
    if bg_image is None:
        print('[WARNING] Background image not found, skipping torso extraction')
        return
    
    image_paths = sorted(glob.glob(os.path.join(ori_imgs_dir, '*.jpg')))
    
    for image_path in tqdm(image_paths, desc="Processing masks"):
        ori_image = cv2.imread(image_path)
        seg_path = image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        seg = cv2.imread(seg_path)
        
        if seg is None:
            continue
        
        # Parse semantic labels
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)
        
        # Save face mask
        face_mask_img = np.zeros_like(seg)
        face_mask_img[head_part] = 255
        cv2.imwrite(image_path.replace('ori_imgs', 'face_mask').replace('.jpg', '.png'), face_mask_img)
        
        # Create GT image (with background filled)
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)
        
        # Create torso image
        torso_image = gt_image.copy()
        torso_image[head_part] = bg_image[head_part]
        
        h, w = torso_image.shape[:2]
        torso_alpha = 255 * np.ones((h, w, 1), dtype=np.uint8)
        
        # Inpainting for torso-head boundary
        neck_part_dilated = binary_dilation(neck_part, iterations=3)
        mask = neck_part | torso_part | neck_part_dilated
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0
        
        cv2.imwrite(
            image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'),
            np.concatenate([torso_image, torso_alpha], axis=-1)
        )
    
    print(f'[INFO] Torso and mask extraction complete')


def save_transforms(base_dir: str, ori_imgs_dir: str):
    """Save camera transforms to JSON format."""
    print(f'[INFO] Saving transforms')
    
    from data_utils.face_tracking.util import euler2rot
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0])
    h, w = tmp_image.shape[:2]
    
    params_path = os.path.join(base_dir, 'bundle_adjustment.pt')
    if not os.path.exists(params_path):
        print(f'[WARNING] Bundle adjustment results not found at {params_path}')
        return
    
    params_dict = torch.load(params_path)
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans']
    valid_num = euler_angle.shape[0]
    
    # Train/val split
    train_val_split = int(valid_num * 10 / 11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)
    
    # Compute rotation matrices
    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
    
    pose = torch.eye(4, dtype=torch.float32)
    
    for split_name, split_ids in [('train', train_ids), ('val', val_ids)]:
        transform_dict = {
            'focal_len': float(focal_len[0]),
            'cx': float(w / 2.0),
            'cy': float(h / 2.0),
            'frames': []
        }
        
        for i in split_ids:
            i = i.item()
            frame_dict = {
                'img_id': i,
                'aud_id': i,
            }
            
            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]
            frame_dict['transform_matrix'] = pose.numpy().tolist()
            
            transform_dict['frames'].append(frame_dict)
        
        output_path = os.path.join(base_dir, f'transforms_{split_name}.json')
        with open(output_path, 'w') as f:
            json.dump(transform_dict, f, indent=2)
    
    print(f'[INFO] Transforms saved')


def prepare_st_gauss_data(video_path: str, output_dir: str = None, skip_steps: list = None):
    """
    Full ST-Gauss data preparation pipeline.
    
    Uses SyncTalk's preprocessing for:
    - Bundle adjustment (Head-Sync Stabilizer)
    - 52 ARKit blendshapes
    - Better semantic segmentation
    """
    if skip_steps is None:
        skip_steps = []
    
    base_dir = output_dir or os.path.dirname(video_path)
    
    # Create directories
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    face_mask_dir = os.path.join(base_dir, 'face_mask')
    flow_dir = os.path.join(base_dir, 'flow_result')
    
    for d in [ori_imgs_dir, parsing_dir, gt_imgs_dir, torso_imgs_dir, face_mask_dir, flow_dir]:
        os.makedirs(d, exist_ok=True)
    
    wav_path = os.path.join(base_dir, 'aud.wav')
    
    # Step 1: Extract audio
    if 'audio' not in skip_steps:
        extract_audio(video_path, wav_path)
    
    # Step 2: Extract frames
    if 'frames' not in skip_steps:
        extract_frames(video_path, ori_imgs_dir)
    
    # Step 3: Extract semantics
    if 'semantics' not in skip_steps:
        extract_semantics(ori_imgs_dir, parsing_dir)
    
    # Step 4: Extract background
    if 'background' not in skip_steps:
        extract_background(base_dir, ori_imgs_dir)
    
    # Step 5: Extract torso and masks
    if 'torso' not in skip_steps:
        extract_torso_and_masks(base_dir, ori_imgs_dir)
    
    # Step 6: Extract landmarks
    if 'landmarks' not in skip_steps:
        extract_landmarks(ori_imgs_dir)
    
    # Step 7: Face tracking
    if 'tracking' not in skip_steps:
        run_face_tracking(ori_imgs_dir, base_dir)
    
    # Step 8: Bundle adjustment (SyncTalk's Head-Sync Stabilizer)
    if 'bundle' not in skip_steps:
        run_bundle_adjustment(base_dir, ori_imgs_dir, face_mask_dir, flow_dir)
    
    # Step 9: Blendshape capture (52 ARKit parameters)
    if 'blendshape' not in skip_steps:
        run_blendshape_capture(base_dir)
    
    # Step 10: Save transforms
    if 'transforms' not in skip_steps:
        save_transforms(base_dir, ori_imgs_dir)
    
    print(f'\n[INFO] ST-Gauss data preparation complete!')
    print(f'Output directory: {base_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-Gauss Data Preparation')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--skip', nargs='+', default=[], 
                        help='Steps to skip: audio, frames, semantics, background, torso, landmarks, tracking, bundle, blendshape, transforms')
    
    args = parser.parse_args()
    
    prepare_st_gauss_data(
        video_path=args.video_path,
        output_dir=args.output_dir,
        skip_steps=args.skip
    )


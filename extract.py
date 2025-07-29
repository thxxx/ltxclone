import os
import numpy as np
import torch
from torchvision.io import read_video
from einops import rearrange
from tqdm import tqdm
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from torch.cuda.amp import autocast
import torch.nn.functional as F

device = 'cuda'
dir_path = '/workspace/AVE_Dataset/AVE_processed/'
pro_path = '/workspace/AVE_Dataset/AVE_latents/'
files = sorted(os.listdir(dir_path))  # 정렬 권장 (일관된 순서)
os.makedirs(pro_path, exist_ok=True)
print("길이 : ", len(files))

ltxv_model_path = './ltxv-2b-0.9.8-distilled.safetensors'
vae = CausalVideoAutoencoder.from_pretrained(ltxv_model_path)
vae.to(device)
vae.eval()

batch_size = 2
buffer = []
file_buffer = []

for fp in tqdm(files):
    vid_path = os.path.join(dir_path, fp)
    video_frames, _, _ = read_video(vid_path)

    # 영상 전처리
    video_frames = rearrange(video_frames, 't h w c -> 1 c t h w').tile(2, 1, 1, 1, 1)  # (1, C, T, H, W)
    video_frames = F.pad(video_frames, (0, 0, 0, 0, 0, 1))
    video_frames = video_frames.to(torch.float32) / 255.0 * 2 - 1
    video_frames = video_frames.to(device).to(torch.bfloat16)

    try:
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                latent = vae.encode(video_frames).latent_dist.mode().float().cpu().numpy()  # (B, ...)
        np.save(os.path.join(pro_path, fp[:-4] + ".npy"), latent[0])
    except Exception as e:
        print(f"\n\n ERROR : {e} \n\n")

    # buffer.append(video_frames)
    # file_buffer.append(fp)

    # # 배치 처리
    # if len(buffer) == batch_size:
    #     batch = torch.cat(buffer, dim=0)  # (B, C, T, H, W)
    #     with torch.no_grad():
    #         with autocast(dtype=torch.bfloat16):
    #             latent = vae.encode(batch).latent_dist.mode().float().cpu().numpy()  # (B, ...)
    #     for i, filename in enumerate(file_buffer):
    #         np.save(os.path.join(pro_path, filename[:-4] + ".npy"), latent[i])
    #     buffer = []
    #     file_buffer = []

# 남은 데이터 처리
# if buffer:
#     batch = torch.cat(buffer, dim=0)
#     with torch.no_grad():
#         with autocast(dtype=torch.bfloat16):
#             latent = vae.encode(batch).latent_dist.mode().float().cpu().numpy()
#     for i, filename in enumerate(file_buffer):
#         np.save(os.path.join(pro_path, filename[:-4] + ".npy"), latent[i])
    
    #     timestep = torch.ones(video_frames.shape[0], device=device) * 0.1
    #     reconstructed_videos = vae.decode(
    #         latent, target_shape=video_frames.shape, timestep=timestep
    #     ).sample
    
    # output_path = './reencoded_video12816test_2.mp4'
    # recon_videos = (rearrange(reconstructed_videos, "b c t h w -> b t h w c")[0].cpu()/2 + 0.5) * 255
    
    # write_video(
    #     filename=output_path,
    #     video_array=recon_videos,      # shape: (T, H, W, C)
    #     fps=int(info['video_fps']),
    #     video_codec='libx264',         # optional
    #     options={"crf": "18"}          # optional: quality setting
    # )
import hydra
import os
import os.path as osp
import numpy as np
import cv2
import logging
import torch
import json
import time

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_monodepth
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
    token_merging_method = hydra_cfg.pi3.get("token_merging_method", "none")
    model = Pi3.from_pretrained(
        pretrained_model_name_or_path,
        enable_token_merging=token_merging_method != "none",
        token_merging_method=token_merging_method,
        token_merging_ratio=hydra_cfg.pi3.get("token_merging_ratio", 0.9),
        token_merging_frame_alpha=hydra_cfg.pi3.get("token_merging_frame_alpha", 0.1),
        token_merging_frame_segment_threshold=hydra_cfg.pi3.get("token_merging_frame_segment_threshold", 0.9),
        token_merging_frame_merge_threshold=hydra_cfg.pi3.get("token_merging_frame_merge_threshold", 0.1),
        token_merging_frame_max_window=hydra_cfg.pi3.get("token_merging_frame_max_window", 20),
        token_merging_frame_pool_stride=hydra_cfg.pi3.get("token_merging_frame_pool_stride", 2),
        token_merging_frame_multi_max_group_size=hydra_cfg.pi3.get("token_merging_frame_multi_max_group_size", 4),
        token_merging_frame_multi_pair_threshold=hydra_cfg.pi3.get("token_merging_frame_multi_pair_threshold", 0.95),
        token_merging_frame_multi_span_threshold=hydra_cfg.pi3.get("token_merging_frame_multi_span_threshold", 0.93),
    ).to(hydra_cfg.device).eval()
    logger = logging.getLogger("monodepth-infer")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            # some datasets (like nyu-v2) have only a set of images, only for monodepth
            seq_list = [None]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        # 3. infer for each sequence
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering monodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            max_frames = int(getattr(hydra_cfg, "max_frames_per_seq", 0) or 0)
            if max_frames > 0:
                filelist = filelist[:max_frames]
            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")

            # 3.2 infer for each image
            total_infer_time = 0.0
            processed_frames = 0
            for file in tqdm(filelist):
                # 3.2.1 skip if the file already exists
                npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                    continue

                # 3.2.2 infer the depth map
                if torch.cuda.is_available() and str(hydra_cfg.device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                depth_map = infer_monodepth(file, model, hydra_cfg)
                if torch.cuda.is_available() and str(hydra_cfg.device).startswith("cuda"):
                    torch.cuda.synchronize()
                total_infer_time += time.perf_counter() - start
                processed_frames += 1

                # 3.2.3 save the depth map to the save_dir as npy
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                elif not isinstance(depth_map, np.ndarray):
                    raise ValueError(f"Unknown depth map type: {type(depth_map)}")
                np.save(npy_save_path, depth_map)

                # 3.2.4 also save the png
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(png_save_path, depth_map)
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump(
                    {
                        "time": total_infer_time,
                        "frames": processed_frames,
                        "fps": processed_frames / total_infer_time if total_infer_time > 0 else 0.0,
                        "token_merging_method": token_merging_method,
                    },
                    f,
                    indent=4,
                )
        # for each dataset
        logger.info(f"Monodepth inference for dataset {dataset_name} finished!")

    del model
    torch.cuda.empty_cache()
    logger.info(f"Monodepth inference for Pi3 finished!")

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()

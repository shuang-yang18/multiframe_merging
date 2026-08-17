import os
import os.path as osp
import logging
import numpy as np
import torch
import hydra
import time

from tqdm import tqdm
from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_cameras_c2w
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg, write_csv, save_list_of_matrices
from relpose.evo_utils import calculate_averages, load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):

    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-distance.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/relpose-angular.yaml

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
    logger = logging.getLogger(f"relpose-dist")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        seq_list = get_all_sequences(dataset_info)
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)

        # 3. infer for each sequence
        model = model.eval()
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering relpose(c2w) on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        results = []
        all_times = []
        all_frames = []
        frame_merge_stats = []
        token_merging_stats = []
        tbar = tqdm(seq_list, desc=f"[{dataset_name} eval]")
        for seq in tbar:
            # 4.1 list all images of this sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            filelist = filelist[:: hydra_cfg.pose_eval_stride]
            max_frames = int(getattr(hydra_cfg, "max_frames_per_seq", 0) or 0)
            if max_frames > 0:
                filelist = filelist[:max_frames]

            # 4.2 real inference
            # pr_poses: c2w poses, (N, 3, 4), in torch
            # pr_intrs: focals + pps, (N, 3, 3), in numpy
            if torch.cuda.is_available() and str(hydra_cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            pr_poses, pr_intrs = infer_cameras_c2w(filelist, model, hydra_cfg)
            if torch.cuda.is_available() and str(hydra_cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            all_times.append(elapsed)
            all_frames.append(len(filelist))
            frame_merge_stats.extend(getattr(model, "last_frame_merge_stats", []))
            token_merging_stats.extend(getattr(model, "last_token_merging_stats", []))
            pred_traj = get_tum_poses(pr_poses)

            # 4.3 save predicted poses & intrinsics
            seq_save_dir = osp.join(output_root, seq)
            os.makedirs(seq_save_dir, exist_ok=True)
            # save predicted poses
            save_tum_poses(pred_traj, osp.join(output_root, seq, "pred_traj.txt"), verbose=hydra_cfg.verbose)
            np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
            save_list_of_matrices(pr_poses.numpy().tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))
            # save predicted intrinsics (if available)
            if pr_intrs is not None:
                np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
                save_list_of_matrices(pr_intrs.tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))

            # 4.4 read ground truth trajectory
            try:
                gt_traj = load_traj(
                    gt_traj_file = dataset_info.anno.path.format(seq=seq),
                    traj_format  = dataset_info.anno.format,
                    stride       = hydra_cfg.pose_eval_stride,
                    num_frames   = len(filelist),
                )
            except np.linalg.LinAlgError:
                logger.warning(f"Failed to load ground truth trajectory for sequence {seq} in dataset {dataset_name}.")
                continue

            # 4.5 evaluate predicted trajectory with ground truth trajectory, plot the trajectory
            if gt_traj is not None:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj,
                    seq      = seq,
                    filename = osp.join(output_root, seq, "eval_metric.txt"),
                    verbose  = hydra_cfg.verbose,
                )
                plot_trajectory(pred_traj, gt_traj, title=seq, filename=osp.join(output_root, seq, "vis.png"), verbose=hydra_cfg.verbose)
            else:
                raise ValueError(f"Ground truth trajectory not found for sequence {seq} in dataset {dataset_name}.")

            # 4.6 save sequence metrics to csv
            seq_metrics = {
                "dataset": dataset_name,
                "seq": seq,
                "ATE": ate,
                "RPE trans": rpe_trans,
                "RPE rot": rpe_rot,
                "fps": len(filelist) / elapsed if elapsed > 0 else 0.0,
                "time": elapsed,
                "frames": len(filelist),
            }
            write_csv(osp.join(output_root, "seq_metrics.csv"), seq_metrics)
            results.append((seq, ate, rpe_trans, rpe_rot))

            # 4.7. update metric for a sequence to tqdm bar
            tbar.set_postfix_str(f"Seq {seq} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")

        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

        dataset_metrics = {
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
            "fps": sum(all_frames) / sum(all_times) if sum(all_times) > 0 else 0.0,
            "total_time": sum(all_times),
            "frames": sum(all_frames),
            "token_merging_method": token_merging_method,
        }
        if frame_merge_stats:
            dataset_metrics["frame_merge_active_frames_mean"] = float(np.mean([s["active_frames_mean"] for s in frame_merge_stats]))
            dataset_metrics["frame_merge_merge_ratio_mean"] = float(np.mean([s["merge_ratio_mean"] for s in frame_merge_stats]))
        if token_merging_stats:
            dataset_metrics["token_merging_full_attention_token_ratio_mean"] = float(np.mean([s["full_attention_token_ratio"] for s in token_merging_stats]))
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, dataset_metrics)
        logger.info(f"{dataset_name} - Average pose estimation metrics: {dataset_metrics}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-distance")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()

import os
import sys
import json
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.models import get_pipeline_by_name
from src.utils.utils import set_seed
from src.data.prompts import make_loader
from src.optim.cmaes import CMAESTrainer
import src.metrics.rewards as rewards

CONFIG = config_flags.DEFINE_config_file("config", default="configs/calibri.py:cmaes_hpsv3_flux_gates",
                                         help_string="Training configuration to reuse for eval")
flags.DEFINE_string("checkpoint_path", None, "Path to the .json checkpoint file.", required=True)
flags.DEFINE_string("save_dir", "./outputs/inference_results", "Directory to save generated images.")
flags.DEFINE_string("prompt", None, "A single text prompt for generation. Overrides dataset evaluation if provided.")

FLAGS = flags.FLAGS

def main(_):
    accelerator = Accelerator()
    cfg = CONFIG.value
    
    if getattr(cfg.experiment, "seed", None) is not None:
        set_seed(cfg.experiment.seed)

    cfg.device = str(accelerator.device)

    if FLAGS.prompt is not None:
        do_metrics = False
        val_loader = DataLoader([FLAGS.prompt], batch_size=1, shuffle=False)
        dataset_name = f'prompt: "{FLAGS.prompt}"'
    else:
        do_metrics = True
        dataset_name = cfg.data.val_dataset
        val_loader = make_loader(
            cfg.data.val_dataset, 
            getattr(cfg.data, "batch_size_val", 4), 
            0, 
            False, 
            False, 
            limit=getattr(cfg.data, "limit_val", -1), 
        )

    infer_dtype = torch.float32
    if getattr(cfg.model, "dtype", "fp32") == "fp16":
        infer_dtype = torch.float16
    elif getattr(cfg.model, "dtype", "fp32") == "bf16":
        infer_dtype = torch.bfloat16

    if accelerator.is_main_process:
        print(f"Initializing pipeline for {cfg.model.model_name}...")
        
    pipeline_class = get_pipeline_by_name(cfg.model.model_name)
    pipeline = pipeline_class(
        device=cfg.device,
        dtype=infer_dtype,
        model_name=cfg.model.model_name,
        num_models=cfg.scaleguidance.num_models,
        verbose=False
    )
    if hasattr(pipeline, "pipeline") and hasattr(pipeline.pipeline, "set_progress_bar_config"):
        pipeline.pipeline.set_progress_bar_config(disable=True)

    eval_reward_fn = None
    if do_metrics:
        scoredict = getattr(cfg, "reward_fn_eval", None) or getattr(cfg, "reward_fn", None)
        if scoredict:
            eval_reward_fn = rewards.multi_score(cfg.device, scoredict)

    if accelerator.is_main_process:
        print(f"Loading weights from {FLAGS.checkpoint_path}...")
        
    with open(FLAGS.checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    sol = np.asarray(payload["solution"], dtype=np.float64)

    trainer = CMAESTrainer(
        cfg, 
        pipeline, 
        reward_fn=None, 
        eval_reward_fn=eval_reward_fn,
        writer=None, 
        train_loader=None, 
        val_loader=val_loader,
        accelerator=accelerator
    )

    if accelerator.is_main_process:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"\nRunning generation {'with metrics' if do_metrics else 'without metrics'} on {dataset_name}...")
        
    vals = trainer._eval_validation(
        sol, 
        seed=getattr(cfg.experiment, "seed", 42),
        save_dir=FLAGS.save_dir,
        save_images=True
    )

    if accelerator.is_main_process and vals and do_metrics:
        print("\n================ Metrics ================")
        for k, v in vals.items():
            print(f"{k}: {v:.4f}")
        print("=========================================\n")

if __name__ == "__main__":
    app.run(main)
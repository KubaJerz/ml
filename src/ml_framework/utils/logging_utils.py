from pathlib import Path
import json
import torch
from typing import Dict

def save_metrics(metrics, name, save_dir):
    Path(save_dir / "metrics").mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics" / f"metrics_{name}.json"
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_model(model, metrics, name, save_dir, save_full_model):
    Path(save_dir / "models").mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "models" / f"{name}.pth"

    if save_full_model:
        torch.save(model, model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    save_metrics(metrics=metrics, name=name, save_dir=save_dir)


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import os
import json
from pathlib import Path
import sys
import torch
import importlib
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ..data_script.EEGDataScript import EEGDataScript
import yaml

class SingleModelEvaluator:
    def __init__(self, evaluation_dir: str):
        self.evaluation_dir = Path(evaluation_dir)
        self._validate_evaluation_dir()
        
    def evaluate(self):
        trial_config = self._load_trial_config()
        metrics = self._load_metrics()
        best_f1_cf = self._compute_confusion_matrix('best_dev_f1', trial_config)
        best_loss_cf = self._compute_confusion_matrix('best_dev_loss', trial_config)
        
        self._create_evaluation_plot(metrics, best_f1_cf, best_loss_cf)

    def _validate_evaluation_dir(self) -> None:
        if not self.evaluation_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {self.evaluation_dir}")
        
        if not self.evaluation_dir.is_dir():
            raise NotADirectoryError(f"Specified path is not a directory: {self.evaluation_dir}")
        
        required_files = [
            'metrics/metrics_best_dev_f1.json',
            'metrics/metrics_best_dev_loss.json',
            'metrics/metrics_full.json',
            'models/best_dev_f1.pth',
            'models/best_dev_loss.pth',
            'models/full.pth'
        ]
        
        for pattern in required_files:
            if not list(self.evaluation_dir.glob(pattern)):
                raise FileNotFoundError(f"Required file matching {pattern} not found in {self.evaluation_dir}")

    def _load_metrics(self) -> Dict[str, Any]:
        metrics_file = self.evaluation_dir / 'metrics' / 'metrics_full.json'
        
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metrics from {metrics_file}: {e}")

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        try:
            obj = torch.load(model_path)
            
            # Case 1: State dictionary format
            if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
                print(f"Loading model from state dict: {model_path.name}")
                config = self._load_trial_config()
                model = self._create_model_from_config(config)
                model.load_state_dict(obj)
                return model
            
            # Case 2: Full model instance
            elif hasattr(obj, "state_dict"):
                print(f"Loading full model instance: {model_path.name}")
                return obj
            
            else:
                raise ValueError(f"Unrecognized model format in {model_path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
            
    def _load_trial_config(self) -> Dict[str, Any]:
        config_file = self.evaluation_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Training config not found: {config_file}")
            
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_model_from_config(self, config: Dict[str, Any]) -> torch.nn.Module:
        model_path = Path(config['model']['absolute_path'])
        
        try:
            dir_path, file_name = os.path.split(model_path)
            module_name = os.path.splitext(file_name)[0]
            
            sys.path.append(str(dir_path))
            model_module = importlib.import_module(module_name)
            model_class = getattr(model_module, module_name.upper())
            
            model_params = {
                'input_size': config['data']['input_size'],
                'input_channels': config['data']['input_channels'],
                'output_size': config['data']['output_size'],
                'num_classes': config['data']['num_classes'],
                'hyperparams': config['parameters']
            }
            
            return model_class(**model_params)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create model from config: {e}")

    def _compute_confusion_matrix(self, model_name, trial_config):
        data_config = trial_config['data']
        
        model_path = self.evaluation_dir / 'models' / f'{model_name}.pth'
        model = self._load_model(model_path)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Setup test data
        data_script = EEGDataScript(data_config)
        _, test_dataset = data_script.get_datasets()
        
        batch_size = len(test_dataset) if data_config['dev_batch_size'] == -1 else data_config['dev_batch_size']
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                if len(y_batch.shape) > 1:  # Handle one-hot encoded labels
                    y_batch = torch.argmax(y_batch, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                else:
                    pred = (torch.sigmoid(outputs) > 0.5).long()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        
        return confusion_matrix(targets, predictions)

    def _create_evaluation_plot(self, metrics: Dict[str, Any], best_f1_cf: np.ndarray, best_loss_cf: np.ndarray) -> None:
        """Create and save the evaluation plot with metrics and confusion matrices."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot Loss
        self._plot_metric(
            ax=axes[0, 0],
            train_data=metrics['train_loss'],
            val_data=metrics['dev_loss'],
            best_value=metrics['best_dev_loss'],
            title='Loss vs Epochs',
            ylabel='Loss',
            best_label=f'Best Dev Loss: {metrics["best_dev_loss"]:.3f}'
        )
        
        # Plot F1
        self._plot_metric(
            ax=axes[0, 1],
            train_data=metrics['train_f1'],
            val_data=metrics['dev_f1'],
            best_value=metrics['best_dev_f1'],
            title='F1 Score vs Epochs',
            ylabel='F1',
            best_label=f'Best Dev F1: {metrics["best_dev_f1"]:.3f}'
        )
        
        # Plot Confusion Matrices
        ConfusionMatrixDisplay(best_loss_cf).plot(ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix for Best Loss Model')
        
        ConfusionMatrixDisplay(best_f1_cf).plot(ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix for Best F1 Model')
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "_singleEval.png")
        plt.close()

    def _plot_metric(self, ax: plt.Axes, train_data: list, val_data: list, best_value: float,
                    title: str, ylabel: str, best_label: str) -> None:
        """Helper method to plot training metrics."""
        ax.plot(train_data, label=f'Train {ylabel}')
        ax.plot(val_data, label=f'Validation {ylabel}')
        ax.axhline(best_value, color='g', linestyle='--', label=best_label)
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Evaluate single model training run")
#     parser.add_argument("eval_dir", type=str, help="Directory containing model evaluation results")
#     parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for testing (-1 for full batch)")
#     parser.add_argument("--random_state", type=int, default=69, help="Random state for dataset split")
    
#     args = parser.parse_args()
    
#     trial_config = {
#         'batch_size': args.batch_size,
#         'random_state': args.random_state
#     }
    
#     evaluator = SingleModelEvaluator(args.eval_dir)
#     evaluator.evaluate(trial_config)
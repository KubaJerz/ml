import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path

class MultiModelEvaluator:
    def __init__(self, evaluation_dir: str):
        self.evaluation_dir = Path(evaluation_dir)
        self._validate_evaluation_dir()
        
    def evaluate(self) -> None:
        fig, axes = self._setup_plot_layout()
        
        subdirectories = self._get_model_subdirectories()
        colors = self._generate_colors(len(subdirectories))
        model_metrics = self._collect_model_metrics(subdirectories, colors)
        
        self._plot_training_curves(axes, model_metrics)
        self._create_summary_tables(axes, model_metrics)
        
        self._finalize_and_save_plot(fig)

    def _validate_evaluation_dir(self) -> None:
        if not self.evaluation_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {self.evaluation_dir}")
        
        if not self.evaluation_dir.is_dir():
            raise NotADirectoryError(f"Specified path is not a directory: {self.evaluation_dir}")
            
        if not any(self.evaluation_dir.iterdir()):
            raise ValueError(f"Evaluation directory is empty: {self.evaluation_dir}")

    def _setup_plot_layout(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 2, figsize=(30, 20))
        
        # init loss plot
        axes[0, 0].set_title('Loss vs Epochs')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        
        # init f1 score plot
        axes[0, 1].set_title('F1 Score vs Epochs')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('F1')
        
        return fig, axes

    def _get_model_subdirectories(self) -> List[Path]:
        return [d for d in self.evaluation_dir.iterdir() if d.is_dir()]

    def _generate_colors(self, num_colors: int) -> List[str]:
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", 
            "#800000", "#008000", "#000080", "#808000", "#800080", "#008080", "#808080", 
            "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0", 
            "#400000", "#004000", "#000040", "#404000", "#400040", "#004040", "#404040", 
            "#200000", "#002000", "#000020", "#202000", "#200020", "#002020", "#202020", 
            "#600000", "#006000", "#000060", "#606000", "#600060", "#006060", "#606060", 
            "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0", "#00A0A0", "#A0A0A0", 
            "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0", "#E0E0E0",
        ]
        return colors[:num_colors]

    def _collect_model_metrics(self, subdirectories: List[Path], colors: List[str]) -> Dict[str, Dict[str, Any]]:
        metrics = {}
        for idx, subdir in enumerate(subdirectories):
            try:
                model_metrics = self._load_metrics(subdir)
                metrics[subdir.name] = {
                    'metrics': model_metrics,
                    'color': colors[idx],
                }
            except Exception as e:
                print(f"Warning: Could not load metrics from {subdir}: {e}")
        return metrics

    def _load_metrics(self, model_dir: Path) -> Dict[str, Any]:
        metrics_file = model_dir / "metrics" / "metrics_full.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
            
        with open(metrics_file, 'r') as f:
            return json.load(f)

    def _plot_training_curves(self, axes: np.ndarray, model_metrics: Dict[str, Dict[str, Any]]) -> None:
        for model_name, data in model_metrics.items():
            metrics = data['metrics']
            color = data['color']
            
            # Plot loss curves
            axes[0, 0].plot(metrics['train_loss'], label=f'Train Loss {model_name}', color=color)
            axes[0, 0].plot(metrics['dev_loss'], label=f'Validation Loss {model_name}', color=color, linestyle='--')
            
            # Plot F1 curves
            axes[0, 1].plot(metrics['train_f1'], label=f'Train F1 {model_name}', color=color)
            axes[0, 1].plot(metrics['dev_f1'], label=f'Validation F1 {model_name}', color=color, linestyle='--')

    def _create_summary_tables(self, axes: np.ndarray, model_metrics: Dict[str, Dict[str, Any]], top_n: int = 2) -> None:
        self._create_metric_table(axes[1, 0], model_metrics, 'best_dev_loss', 'Best Dev Loss', top_n, higher_better=False)
        self._create_metric_table(axes[1, 1], model_metrics, 'best_dev_f1', 'Best Dev F1', top_n, higher_better=True)

    def _create_metric_table(self, ax: plt.Axes, model_metrics: Dict[str, Dict[str, Any]], 
                           metric_key: str, title: str, top_n: int, higher_better: bool) -> None:
        sorted_models = sorted(
            model_metrics.items(),
            key=lambda x: x[1]['metrics'][metric_key],
            reverse=higher_better
        )[:top_n]

        data = []
        cell_colors = []
        text_colors = []
        
        for model_name, model_data in sorted_models:
            value = model_data['metrics'][metric_key]
            data.append([f"{model_name}: {value:.3f}"])
            cell_color = model_data['color']
            cell_colors.append([cell_color])
            # Use white text for dark backgrounds, black for light backgrounds
            luminance = np.mean(plt.matplotlib.colors.to_rgb(cell_color))
            text_colors.append(['white' if luminance < 0.5 else 'black'])

        table = ax.table(
            cellText=data,
            rowLabels=range(1, top_n + 1),
            colLabels=[f'{title} ----'],
            loc='upper center',
            cellColours=cell_colors
        )

        for (i, j), cell in table._cells.items():
            if i > 0:  # Skip header row
                cell.get_text().set_color(text_colors[i-1][0])
                
        table.auto_set_font_size(False)
        table.set_fontsize(25)
        table.scale(1, 3)
        
        ax.axis('off')
        ax.set_title(f'Top {top_n} Models by {title}', fontsize=30)

    def _finalize_and_save_plot(self, fig: plt.Figure) -> None:
        fig.legend(loc='center right', bbox_to_anchor=(1.12, 0.7), fontsize='x-large')
        plt.tight_layout()
        
        output_path = self.evaluation_dir / "_multiEval.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multiple model training runs")
    parser.add_argument("eval_dir", type=str, help="Directory containing model evaluation results")
    args = parser.parse_args()
    
    evaluator = MultiModelEvaluator(args.eval_dir)
    evaluator.evaluate()
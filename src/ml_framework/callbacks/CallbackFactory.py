from typing import Dict, List, Any
import logging
from . import *

logger = logging.getLogger(__name__)

class CallbackFactory():
    default_configs = {
        'plot_combined_metrics': True,
        'best_dev_f1': True,
        'best_dev_loss': True,
        'early_stopping': False,
        'training_completion_save': True
    }

    @classmethod
    def setup_callbacks(cls, config: Dict[str, Any], metrics: Dict[str, Any]) -> List[Any]:
        callbacks = []

        for callback_name, default_enabled in cls.default_configs.items():
            try:
                is_enabled = config.get(callback_name, default_enabled)
                
                if is_enabled:
                    callback = cls._create_callback(callback_name, config, metrics)
                    if callback:
                        callbacks.append(callback)
                        logger.debug(f"Added {callback_name} callback")

            except Exception as e:
                logger.error(f"Failed to setup {callback_name} callback: {str(e)}")
                continue

        return callbacks

    @classmethod
    def _create_callback(cls, callback_name: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> Any:
        try:
            if callback_name == 'early_stopping':
                best_val_so_far = metrics[f"best_{config.get('early_stopping_monitor', 'dev_loss')}"]
                return EarlyStoppingCallback.EarlyStoppingCallback(
                    best_val_so_far=best_val_so_far,
                    patience=config.get('early_stopping_patience', 10),
                    monitor=config.get('early_stopping_monitor', 'dev_loss'),
                    min_delta=config.get('min_delta', 0)
                )
                
            elif callback_name == 'best_dev_f1':
                return BestMetricCallback.BestMetricCallback(
                    best_value=metrics['best_dev_f1'],
                    metric_to_monitor='dev_f1'
                )
                
            elif callback_name == 'best_dev_loss':
                return BestMetricCallback.BestMetricCallback(
                    best_value=metrics['best_dev_loss'],
                    metric_to_monitor='dev_loss'
                )
                
            elif callback_name == 'plot_combined_metrics':
                return PlotCombinedMetrics.PlotCombinedMetrics(
                    plot_live=config.get('plot_metrics_live', True)
                )
                
            elif callback_name == 'training_completion_save':
                return TrainingCompletionCallback.TrainingCompletionCallback()
                
            else:
                logger.warning(f"Unknown callback type: {callback_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {callback_name} callback: {str(e)}")
            return None
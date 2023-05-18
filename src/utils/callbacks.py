import logging
import pprint
import shutil
import time
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

from transformers import TrainerCallback

logger = getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class LoggingCallback(TrainerCallback):
    """A `TrainerCallback` for saving weights periodically."""
    def __init__(self, save_interval: float):
        """
        Args:
            save_interval (float): An interval to save weights in seconds.  
        """
        self.save_interval = save_interval
        self.start_time = time.time()
        self.save_counter = 1

    
    def on_log(self, args, state, control, logs=None, **kwargs):
        current_duration = time.time() - self.start_time
        if (current_duration // (self.save_interval * self.save_counter)) >= 1:
            logger.info(f'Save weights at {state.global_step} steps trained for '
                        f'{self.save_interval} * {self.save_counter} seconds!')
            self.save_counter += 1
            control.should_save = True


class SaveNBestModelCallback(TrainerCallback):
    """A `TrainerCallback` for saving n-best checkpoints."""
    def __init__(self, n_best: int = 5) -> None:
        """__init__

        Args:
            n_best (int, optional): Number of maximum checkpoints to be saved. Defaults to 5.
        """
        super().__init__()
        self.n_best = n_best
        self.n_best_metric_path: List[Tuple(float, str)] = []
        
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)
        
        if metric_value is None:
            logger.warning(
                f"No metric_for_best_model found!"
            )
            return

        # check list
        if len(self.n_best_metric_path) < self.n_best:
            output_dir = args.output_dir
            num_steps = state.global_step
            check_pointpath = Path(output_dir) / 'checkpoint-{}'.format(num_steps)
            self.n_best_metric_path.append(
                (metric_value, str(check_pointpath))
            )
            control.should_save = True
        else:
            self.n_best_metric_path = sorted(self.n_best_metric_path, key=lambda x: x[0])
            (smallest_metric_value, smallest_checkpoint_path) = self.n_best_metric_path[0]
            if metric_value > smallest_metric_value:
                # need to delete the checkpoint with the smallest metric value
                _ = self.n_best_metric_path.pop(0)
                shutil.rmtree(smallest_checkpoint_path)
                print(f'{smallest_checkpoint_path} has been deleted due to max cap!')
                
                # add a current checkpoint
                output_dir = args.output_dir
                num_steps = state.global_step
                check_pointpath = Path(output_dir) / 'checkpoint-{}'.format(num_steps)
                self.n_best_metric_path.append(
                    (metric_value, str(check_pointpath))
                )
                control.should_save = True
                pprint.pprint(f'Current n_best checkpoint list: {self.n_best_metric_path}')


class EarlyStoppingCallback(TrainerCallback):
    """A `TrainerCallback` for enabling a model to early stop its training.
    References: 
        https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback
    """

    def __init__(self, patience: int, metric_name: str, objective_type: str, verbose: bool=True):
        """
        Args:
            patience (int): Patience value for early stopping.
                            Actual resulting value should be (logging_steps * patience).
            metric_name (str): Which metric to watch? The name should be based on `metrics`.
            objective_type (str): `maximize` or `minimize`.
            verbose (bool): Whether to show status. (default=True)
        """
        self.patience = patience
        self.metric_name = metric_name
        self.objective_type = objective_type
        self.verbose = verbose
        self.best_metric = None
        self.counter = 0
        

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Event called after the evaluation phase."""
        
        # score calculation
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        score = metrics.get(metric_to_check)
        logger.info(f"[Early stopping]: Score >>> {score}")
        
        if score is None: # illegal name is given
            return
        if self.objective_type == "minimize": # such as loss
            score = -score

        # score evaluation
        if self.best_metric is None: 
            # init
            self.best_metric = score
            logger.info(f"[Early stopping]: Will Save weights...")
            control.should_save = True

        elif score < self.best_metric: 
            # not improved
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping counter: {self.counter} / {self.patience}")
                logger.info(f"[Early stopping]: Will NOT save weights...")
                control.should_save = False
            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(f"Early stopping applied at {state.global_step} steps!")
                control.should_save = False
                control.should_training_stop = True

        else: # improved
            self.counter = 0
            if self.verbose:
                logger.info(f"Score improved: {self.best_metric:.3f} -> {score:.3f}")
            self.best_metric = score
            state.best_model_checkpoint = args.output_dir + "/checkpoint-" + str(state.global_step)
            logger.info(f"[Early stopping]: Will Save weights...")
            control.should_save = True

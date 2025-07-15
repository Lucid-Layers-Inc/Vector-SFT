import os
import pandas as pd
from transformers import TrainerCallback


class ClearMLCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.logger.report_scalar(title="Training", series=key, value=value, iteration=state.global_step)



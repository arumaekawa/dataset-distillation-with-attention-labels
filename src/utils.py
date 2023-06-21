import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from transformers import BatchEncoding


def average(inputs: list[int | float | list | dict], std: bool = False):
    if isinstance(inputs[0], (int, float)):
        if std:
            return (np.mean(inputs), np.std(inputs))
        else:
            return np.mean(inputs)
    elif isinstance(inputs[0], list):
        return [average([*ls], std=std) for ls in zip(*inputs)]
    elif isinstance(inputs[0], dict):
        return {k: average([dc[k] for dc in inputs], std=std) for k in inputs[0].keys()}
    else:
        raise TypeError


def log_params_from_omegaconf_dict(params):
    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f"{parent_name}.{k}", v)
                else:
                    mlflow.log_param(f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f"{parent_name}.{i}", v)

    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def batch_on_device(batch: dict[str, torch.Tensor] | BatchEncoding):
    return {k: v.cuda() for k, v in batch.items()}


def endless_dataloader(data_loader, max_iteration=1000000):
    for _ in range(max_iteration):
        for batch in data_loader:
            yield batch

    assert False, "Reach max iteration"

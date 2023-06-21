import json
import logging
import os
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class LearnerTrainConfig:
    train_step: int = 1
    batch_size_per_label: int = 1


@dataclass
class DistilledDataConfig:
    pretrained_data_path: Optional[str] = None
    data_per_label: int = 1
    attention_label_type: str = "none"  # ["none", "cls", "all"]
    seq_length: int = 512
    label_type: str = "hard"  # ["hard", "soft", "unrestricted"]
    lr_for_step: bool = True
    lr_init: float = 1.0e-2
    lr_linear_decay: bool = False
    fix_order: bool = True
    attention_loss_lambda: float = 1.0

    def __post_init__(self):
        assert self.label_type in ["hard", "soft", "unrestricted"]
        if self.lr_for_step and self.lr_linear_decay:
            logger.warning("`lr_linear_decay=True` is ignored.")


class DistilledFeature(metaclass=ABCMeta):
    def __init__(self):
        self.data: torch.Tensor

    def initialize_data(self, initialized_values: torch.Tensor, size_strict=True):
        if size_strict:
            assert (
                self.data.shape == initialized_values.shape
            ), f"{self.data.shape} should be matched to {initialized_values.shape}"
        else:
            raise NotImplementedError

        with torch.no_grad():
            self.data.copy_(initialized_values)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def cuda(self):
        if not self.data.is_cuda:
            grad = self.data.grad
            self.data = (
                self.data.detach().cuda().requires_grad_(self.data.requires_grad)
            )
            self.data.grad = grad


class DistilledInputEmbedding(DistilledFeature):
    def __init__(
        self,
        data_per_label: int = 1,
        num_labels: int = 2,
        seq_length: int = 512,
        hidden_size: int = 768,
    ):
        initial_embeds = torch.randn(
            data_per_label * num_labels, seq_length, hidden_size, requires_grad=True
        )
        self.data = initial_embeds

    def __getitem__(self, index):
        return self.data[index]


class DistilledLabel(DistilledFeature):
    def __init__(
        self,
        data_per_label: int = 1,
        num_labels: int = 2,
        label_type: Literal["hard", "soft", "unrestricted"] = "hard",
    ):
        self.label_type = label_type

        label_ids = torch.arange(data_per_label * num_labels) % num_labels
        self.data = torch.eye(num_labels)[label_ids]

        if label_type != "hard":
            self.data.requires_grad_()

    def __getitem__(self, index):
        if self.label_type == "soft":
            return self.data[index].softmax(dim=-1)

        return self.data[index]


class DistilledAttentionLabels(DistilledFeature):
    def __init__(
        self,
        data_per_label: int = 1,
        num_labels: int = 2,
        seq_length: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        attention_label_type: Literal["cls", "all"] = "cls",
    ):
        assert attention_label_type in ["cls", "all"]

        self.data = torch.randn(
            data_per_label * num_labels,
            num_layers,
            num_heads,
            1 if attention_label_type == "cls" else seq_length,
            seq_length,
            requires_grad=True,
        )

    def __getitem__(self, index):
        return self.data[index].softmax(dim=-1)


class DistilledLR(DistilledFeature):
    def __init__(
        self,
        lr_init: float = 1.0e-3,
        lr_for_step: bool = False,
        lr_linear_decay: bool = False,
        train_step: int = 100,
    ):
        self.lr_linear_decay = lr_linear_decay
        self.lr_for_step = lr_for_step
        self.train_step = train_step

        # Inverse conversion of Softplus()
        lr_init_inv = torch.tensor(lr_init).exp().sub(1.0).log()

        if self.lr_for_step:
            self.data = (
                lr_init_inv.unsqueeze(0).expand(train_step).clone().requires_grad_()
            )
        else:
            self.data = lr_init_inv.requires_grad_()

        self.data.requires_grad_()

    def __getitem__(self, index) -> torch.Tensor:
        if self.lr_for_step:
            return F.softplus(self.data[index])

        steps = torch.arange(
            self.train_step, dtype=torch.float, device=self.data.device
        )[index]
        scale = torch.ones_like(steps)
        if self.lr_linear_decay:
            scale.sub_(steps / self.train_step)

        return F.softplus(self.data) * scale


class DistilledData:
    def __init__(
        self,
        config: DistilledDataConfig,
        train_config: LearnerTrainConfig,
        num_labels: int = 2,
        hidden_size: int = 768,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
    ):
        if not isinstance(config, DistilledDataConfig):
            config = DistilledDataConfig(**config)
        self.config = config

        if not isinstance(train_config, LearnerTrainConfig):
            train_config = LearnerTrainConfig(**train_config)
        self.train_config = train_config

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        if self.config.fix_order:
            assert config.data_per_label % train_config.batch_size_per_label == 0

        self.inputs_embeds = DistilledInputEmbedding(
            data_per_label=config.data_per_label,
            num_labels=num_labels,
            seq_length=config.seq_length,
            hidden_size=hidden_size,
        )
        self.labels = DistilledLabel(
            data_per_label=config.data_per_label,
            num_labels=num_labels,
            label_type=config.label_type,
        )
        self.lr = DistilledLR(
            lr_init=config.lr_init,
            lr_for_step=config.lr_for_step,
            lr_linear_decay=config.lr_linear_decay,
            train_step=train_config.train_step,
        )
        self.data: dict[str, DistilledFeature] = {
            "inputs_embeds": self.inputs_embeds,
            "labels": self.labels,
            "lr": self.lr,
        }

        # attention labels
        if config.attention_label_type in ("cls", "all"):
            self.attention_labels = DistilledAttentionLabels(
                data_per_label=config.data_per_label,
                num_labels=num_labels,
                seq_length=config.seq_length,
                num_layers=num_layers,
                num_heads=num_heads,
                attention_label_type=config.attention_label_type,
            )
            self.data["attention_labels"] = self.attention_labels
        else:
            assert config.attention_label_type == "none"
            self.attention_labels = None

        self.attention_loss_lambda = self.config.attention_loss_lambda

    def get_batch(self, step):
        indices = self.get_batch_indices(step)
        return {
            "inputs_embeds": self.inputs_embeds[indices],
            "labels": self.labels[indices],
            "attention_labels": self.attention_labels[indices]
            if self.attention_labels is not None
            else None,
            "lr": self.lr[step],
        }

    def get_batch_indices(self, step):
        batch_size = self.num_labels * self.train_config.batch_size_per_label
        data_size = self.num_labels * self.config.data_per_label
        if self.config.fix_order:
            cycle = step % int(data_size / batch_size)
            return torch.arange(batch_size * cycle, batch_size * (cycle + 1))
        else:
            return torch.randperm(data_size)[:batch_size]

    def data_dict(self, detach: bool = False):
        data_dict = {name: feature.data for name, feature in self.data.items()}
        if detach:
            data_dict = {name: data.detach().cpu() for name, data in data_dict.items()}
        return data_dict

    def save_pretrained(self, save_path: str | os.PathLike):
        os.makedirs(save_path, exist_ok=True)

        # save config as json file
        config = {
            "config": asdict(self.config),
            "train_config": asdict(self.train_config),
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)

        # save distilled data
        torch.save(self.data_dict(detach=True), os.path.join(save_path, "data_dict"))

        logger.info(f"Save distilled data in `{save_path}`")

    def load_data_dict(self, data_dict: dict[str, torch.Tensor]):
        assert (
            self.data.keys() == data_dict.keys()
        ), f"given keys: {self.data.keys()}, expected keys: {data_dict.keys()}"
        for name in self.data.keys():
            self.data[name].initialize_data(data_dict[name])

    @classmethod
    def from_pretrained(cls, save_path: str | os.PathLike):
        assert os.path.exists(save_path)

        # load config
        config = json.load(open(os.path.join(save_path, "config.json")))
        distilled_data = cls(**config)

        # load data
        pretrained_data = torch.load(os.path.join(save_path, "data_dict"))
        distilled_data.load_data_dict(pretrained_data)
        logger.info(f"Load distilled data from `{save_path}`")

        return distilled_data

    def cuda(self):
        for feature in self.data.values():
            feature.cuda()

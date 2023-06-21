from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from data import TASK_ATTRS

AUTO_MODEL_CLASSES = {"single_label_classification": AutoModelForSequenceClassification}

MODEL_ATTRS = {
    "bert-base-uncased": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "initialized_module_names": ["classifier"],
    },
    "roberta-base": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
        "initialized_module_names": ["classifier"],
    },
}


@dataclass
class ModelConfig:
    """Config for Learner Model"""

    task_name: str
    model_name: str = "bert-base-uncased"
    use_pretrained_model: bool = True
    disable_dropout: bool = True

    def __post_init__(self):
        assert self.model_name in MODEL_ATTRS


class LearnerModel(nn.Module):
    def __init__(self, config: ModelConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.problem_type = TASK_ATTRS[self.config.task_name]["problem_type"]
        self.num_labels = num_labels

        assert self.problem_type != "single_label_classification" or self.num_labels > 1

        if self.config.disable_dropout:
            dropout_configs = {
                dropout_key: 0.0
                for dropout_key in MODEL_ATTRS[self.config.model_name]["dropout_keys"]
            }
        else:
            dropout_configs = {}

        self.bert_model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,
            finetuning_task=self.config.task_name,
            problem_type=self.problem_type,
            **dropout_configs,
        )
        model_class = AUTO_MODEL_CLASSES[self.problem_type]
        self.bert_model: PreTrainedModel = model_class.from_pretrained(
            config.model_name,
            from_tf=bool(".ckpt" in config.model_name),
            config=self.bert_model_config,
        )

        if self.config.use_pretrained_model:
            self.initial_state_dict = self.bert_model.state_dict()
            self.initialized_module_names = MODEL_ATTRS[self.config.model_name][
                "initialized_module_names"
            ]

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def forward(self, *args, **kwargs) -> SequenceClassifierOutput:
        labels: torch.LongTensor = kwargs.pop("labels") if "labels" in kwargs else None

        outputs: SequenceClassifierOutput = self.bert_model(*args, **kwargs)

        loss = None
        if labels is not None:
            if self.problem_type != "single_label_classification":
                raise NotImplementedError

            if outputs.logits.shape == labels.shape:
                # labels: (batch_size, num_labels) or (batch_size)
                labels = labels.view(-1, self.num_labels)
            else:
                assert labels.ndim == 1

            loss = F.cross_entropy(
                outputs.logits.view(-1, self.num_labels), labels, reduction="none"
            )
            assert loss.shape == labels.shape[:1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        return self.bert_model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.bert_model.get_input_embeddings()

    def init_weights(self):
        """init_weights
        Initialize additional weights of pretrained model in the same way
        when calling AutoForSequenceClassification.from_pretrained()
        """

        if not self.config.use_pretrained_model:
            assert hasattr(self.bert_model, "init_weights")
            self.bert_model.init_weights()
        else:
            self.bert_model.load_state_dict(self.initial_state_dict)
            for module_name in self.initialized_module_names:
                initialized_module = self.bert_model
                for p in module_name.split("."):
                    initialized_module = getattr(initialized_module, p)
                for module in initialized_module.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(
                            mean=0.0, std=self.bert_model.config.initializer_range
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif len(list(module.parameters(recurse=False))) > 0:
                        raise NotImplementedError

    @property
    def device(self):
        return self.bert_model.device

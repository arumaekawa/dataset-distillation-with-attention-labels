import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, disable_progress_bar, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

disable_progress_bar()


TASK_ATTRS = {
    # AGNEWS
    "ag_news": {
        "load_args": ("ag_news",),
        "sentence_keys": ("text",),
        "problem_type": "single_label_classification",
        "test_split_key": "test",
    },
    # GLUE
    "mrpc": {
        "load_args": ("glue", "mrpc"),
        "sentence_keys": ("sentence1", "sentence2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
    },
    "qnli": {
        "load_args": ("glue", "qnli"),
        "sentence_keys": ("question", "sentence"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
    },
    "sst2": {
        "load_args": ("glue", "sst2"),
        "sentence_keys": ("sentence",),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
    },
    "qqp": {
        "load_args": ("glue", "qqp"),
        "sentence_keys": ("question1", "question2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
    },
    "mnli": {
        "load_args": ("glue", "mnli"),
        "sentence_keys": ("premise", "hypothesis"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation_matched",
    },
}


@dataclass
class DataConfig:
    task_name: str
    datasets_path: Path
    preprocessed_datasets_path: Path
    train_batch_size: int = 32
    valid_batch_size: int = 256
    test_batch_size: int = 256
    num_proc: int = 1
    force_preprocess: bool = False


class DataModule:
    """DataModule class
    ```
    data_module = DataModule(
        config.data,
        tokenizer_generator=generator.tokenizer,
        tokenizer_learner=learner.tokenizer,
    )
    # preprocess datasets
    data_module.run_preprocess(tokenizer=tokenizer)
    # preprocess external dataset (distilled data)
    data_module.preprocess_dataset(tokenizer=tokenizer, dataset=dataset)
    ```
    """

    def __init__(self, config: DataConfig):
        self.config = config

        # load raw dataset
        self.dataset_attr = TASK_ATTRS[self.config.task_name]
        self.datasets: DatasetDict = self.get_dataset()
        logger.info(f"Datasets: {self.datasets}")

        self.num_labels = self.datasets["train"].features["labels"].num_classes

        # preprocessed_dataset
        self.preprocessed_datasets = None

        # data collator
        self.data_collator = None

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        else:
            assert self.config.task_name in TASK_ATTRS
            datasets = load_dataset(*self.dataset_attr["load_args"])

            if "validation" not in datasets:
                datasets["validation"] = datasets.pop(
                    self.dataset_attr["test_split_key"]
                )
            assert datasets.keys() >= {"train", "validation"}

            os.makedirs(os.path.dirname(self.config.datasets_path), exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        if (
            TASK_ATTRS[self.config.task_name]["problem_type"]
            == "single_label_classification"
        ):
            # rename label_key
            assert "label" in datasets["train"].features
            datasets = datasets.rename_column("label", "labels")
        else:
            raise NotImplementedError

        return datasets

    def run_preprocess(self, tokenizer: PreTrainedTokenizerFast):
        """datasets preprocessing"""

        # set data_collator
        if self.data_collator is None:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8
            )

        if (
            os.path.exists(self.config.preprocessed_datasets_path)
            and not self.config.force_preprocess
        ):
            logger.info(
                "Load preprocessed datasets from `{}`".format(
                    self.config.preprocessed_datasets_path
                )
            )
            self.preprocessed_datasets = load_from_disk(
                self.config.preprocessed_datasets_path
            )
            return

        self.preprocessed_datasets = self.preprocess_dataset(
            tokenizer=tokenizer, dataset=self.datasets
        )

        logger.info(
            f"Save preprocessed datasets to `{self.config.preprocessed_datasets_path}`"
        )
        os.makedirs(
            os.path.dirname(self.config.preprocessed_datasets_path), exist_ok=True
        )
        self.preprocessed_datasets.save_to_disk(self.config.preprocessed_datasets_path)

    def preprocess_dataset(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset: Optional[Dataset | DatasetDict],
    ) -> Dataset | DatasetDict:
        # sentence keys for task
        sentence_keys = TASK_ATTRS[self.config.task_name]["sentence_keys"]

        # get tokenize function
        def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
            sentences = [[s.strip() for s in batch[key]] for key in sentence_keys]
            return tokenizer(
                *sentences, max_length=tokenizer.model_max_length, truncation=True
            )

        # tokenize
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Tokenize datasets",
        )

        remove_keys = [
            col
            for col in dataset["train"].column_names
            if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]
        ]
        dataset = dataset.remove_columns(remove_keys)

        return dataset

    def train_loader(self) -> DataLoader:
        assert "train" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["train"],
            batch_size=self.config.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
        )

    def valid_loader(self) -> DataLoader:
        assert "validation" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["validation"],
            batch_size=self.config.test_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=False,
        )

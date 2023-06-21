import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig, Evaluator
from model import LearnerModel, ModelConfig
from trainer import TrainConfig, Trainer
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add main.log
            mlflow.log_artifact(os.path.join(output_dir, "main.log"))
        return out

    return wrapper


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@mlflow_start_run_with_hydra
def main(config: Config):

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # Set seed
    set_seed(config.base.seed)

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data)

    # Learner
    logger.info(f"Building Leaner model: (`{config.model.model_name}`)")
    model = LearnerModel(config.model, num_labels=data_module.num_labels)

    # preprocess datasets
    data_module.run_preprocess(tokenizer=model.tokenizer)

    # Distilled data
    if config.distilled_data.pretrained_data_path is not None:
        distilled_data = DistilledData.from_pretrained(
            config.distilled_data.pretrained_data_path
        )
    else:
        distilled_data = DistilledData(
            config=config.distilled_data,
            train_config=config.learner_train,
            num_labels=data_module.num_labels,
            hidden_size=model.bert_model_config.hidden_size,
            num_layers=model.bert_model_config.num_hidden_layers,
            num_heads=model.bert_model_config.num_attention_heads,
        )

    # Evaluator
    evaluator = Evaluator(config.evaluate, model=model)

    # Train distilled data
    if not config.train.skip_train:
        trainer = Trainer(config.train)
        trainer.fit(
            distilled_data=distilled_data,
            model=model,
            train_loader=data_module.train_loader(),
            valid_loader=data_module.valid_loader(),
            evaluator=evaluator,
        )

    # Evaluate
    results = evaluator.evaluate(
        distilled_data, eval_loader=data_module.valid_loader(), verbose=True
    )
    mlflow.log_metrics({f"avg.{k}": v[0] for k, v in results.items()})
    mlflow.log_metrics({f"std.{k}": v[1] for k, v in results.items()})

    results = {k: f"{v[0]}±{v[1]}" for k, v in results.items()}
    logger.info(f"Final Results: {results}")
    save_path = os.path.join(config.base.save_dir, "results.json")
    json.dump(results, open(save_path, "w"))
    mlflow.log_artifact(save_path)

    return


if __name__ == "__main__":
    main()

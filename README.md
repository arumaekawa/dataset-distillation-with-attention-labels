# Dataset Distillation with Attention Labels

Implementation of **"Dataset Distillation with Attention Labels for fine-tuning BERT"** (accepted by ACL2023 main (short))

Abstract: Dataset distillation aims to create a small dataset of informative synthetic samples to rapidly train neural networks that retain the performance of the original dataset. In this paper, we focus on constructing distilled few-shot datasets for natural language processing (NLP) tasks to fine-tune pre-trained transformers. Specifically, we propose to introduce attention labels, which can efficiently distill the knowledge from the original dataset and transfer it to the transformer models via attention probabilities. We evaluated our dataset distillation methods in four various NLP tasks and demonstrated that it is possible to create distilled few-shot datasets with the attention labels, yielding impressive performances for fine-tuning BERT. Specifically, in AGNews, a four-class news classification task, our distilled few-shot dataset achieved up to 93.2% accuracy, which is 98.5% performance of the original dataset even with only one sample per class and only one gradient step.

Paper: [Aru Maekawa, Naoki Kobayashi, Kotaro Funakoshi, and Manabu Okumura. 2023. Dataset Distillation with Attention Labels for Fine-tuning BERT. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_, pages 119–127, Toronto, Canada. Association for Computational Linguistics.](https://aclanthology.org/2023.acl-short.12/)

Example of Distilled Data: [Google Drive](https://drive.google.com/file/d/10DkcGEfw9DTWuxBQciin0jGyr9yMQC0H/view?usp=sharing)

Demonstration: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13RJSM35HZM4pPwSmCuFAKuaBtXNzU9II?usp=sharing)

## Contents

This repository utilizes [PyTorch](https://pytorch.org/) and modern experiment manager tools, [Hydra](https://hydra.cc/) and [MLflow](https://www.mlflow.org/).

Datasets and pre-trained models are downloaded and used with [Hugging Face](https://huggingface.co/).

Directory structure:

```
.
├── configs
│   └── default.yaml
├── src
│   ├── data.py
│   ├── distilled_data.py
│   ├── evaluator.py
│   ├── main.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run Scripts

1. Clone this repository.
   ```
   $ git clone https://github.com/arumaekawa/dataset-distillation-with-attention-labels
   $ cd dataset-distillation-with-attention-labels
   ```
2. Prepare environment for **Python 3.10** and install requirements.
   ```
   $ pip install -r requirements.txt
   ```
3. Run experiments.
   ```
   $ python src/main.py -m \
      data.task_name=sst2 \
      distilled_data.label_type=unrestricted \
      distilled_data.attention_label_type=cls \
      distilled_data.lr_init=0.01,0.1 \
      train.lr_inputs_embeds=0.1,0.01,0.001
   ```
4. Check results with mlflow.
   ```
   $ mlflow server --backend-store-uri ./mlruns --host 0.0.0.0
   $ open http://localhost:5000
   ```

<!-- ## References: -->

## Citation

```
@inproceedings{maekawa-etal-2023-dataset,
    title = "Dataset Distillation with Attention Labels for Fine-tuning {BERT}",
    author = "Maekawa, Aru  and
      Kobayashi, Naoki  and
      Funakoshi, Kotaro  and
      Okumura, Manabu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.12",
    pages = "119--127",
    abstract = "Dataset distillation aims to create a small dataset of informative synthetic samples to rapidly train neural networks that retain the performance of the original dataset. In this paper, we focus on constructing distilled few-shot datasets for natural language processing (NLP) tasks to fine-tune pre-trained transformers. Specifically, we propose to introduce attention labels, which can efficiently distill the knowledge from the original dataset and transfer it to the transformer models via attention probabilities. We evaluated our dataset distillation methods in four various NLP tasks and demonstrated that it is possible to create distilled few-shot datasets with the attention labels, yielding impressive performances for fine-tuning BERT. Specifically, in AGNews, a four-class news classification task, our distilled few-shot dataset achieved up to 93.2{\%} accuracy, which is 98.5{\%} performance of the original dataset even with only one sample per class and only one gradient step.",
}
```

# transformers_wrapper

I created this package because I want a quick way to test out those state-of-the-art Transformer models.

## Acknowledgements

This project is heavily inspired by Chris McCormick's public tutorial: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

## Installation

`pip install -i https://test.pypi.org/simple/ --no-deps transformers-wrapper`

## Example

### Text classification

#### Preparation

```python
from transformers_wrapper.classification_models import AvailableClassificationModels, TransformerModelForClassification

model = TransformerModelForClassification(
    model_name=AvailableClassificationModels.BERT_BASE_UNCASED,
    num_labels=len(YOUR_DATAFRAME[LABEL_COL].unique()),
    do_lower_case=True
)

param_optimizer = list(model.model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

train, val = train_test_split(df_train, random_state=1234, test_size=0.2)

sentences_train = train[SENTENCE_COL]
labels_train = train[LABEL_COL]

sentences_val = val[SENTENCE_COL]
labels_val = val[LABEL_COL]

training_dataloader = model.prepare_data(sentences_train, labels=labels_train, max_len=128, batch_size=32)
validation_dataloader = model.prepare_data(sentences_val, labels_val, max_len=128, batch_size=32)
```

#### Finetuning

```python
training_loss = model.finetune(training_dataloader, validation_dataloader, optimizer, epochs=6)
```

#### Evaluation

```python
test_dataloader = model.prepare_data(sentences_test, labels=labels_test, max_len=128, batch_size=32)
true_labels, pred_labels = model.evaluate(test_dataloader, code_book)
```

#### Prediction

```python
true_labels, pred_labels = model.evaluate(pred_dataloader)
```

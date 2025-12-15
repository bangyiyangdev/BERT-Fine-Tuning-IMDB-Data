# BERT Model Sentiment Analysis on IMDb Movie Reviews

## Intro
Sentiment analysis is a use case of text classification which consists of assigning a category to a given text. It's a powerful Natural Language Processing (NLP) technique that makes it possible to automatically analyze what people think about a certain topic. This can help companies and individuals to quickly make more informed decisions. Sentiment analysis has for example applications in [social media, customer service and market research](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases).  

## Data
-> Link: IMDB Dataset of 50K Movie Reviews

IMDB dataset having 50K movie reviews for natural language processing or Text analytics. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms. For more dataset information, please go through the following link

## Model and Training
I leveraged the pretrained BERT (Bidirectional Encoder Representations from Transformers) made available by Hugging Face. The model (bert-base-uncased) was fine-tuned using the following hyper-parameters:
* `Learning rate = 2e-5` using `AdamW` optimizer
* `Linear scheduler` with `num_warmup_steps = 0`
* `Maximum sequence length = 128`
* `Batch size = 32`
* `Number of training epochs = 5`

[Here](https://huggingface.co/docs/transformers/training) is a nice tutorial from Hugging Face explaining how to fine-tune a pretrained model.

The model is quite heavy (approx. 427 MB). So after fine-tuning, I pushed it to the [Hugging Face hub](https://huggingface.co/MLphile/fine_tuned_bert-movie_review), from where it can be accessed using the following checkpoint address: `MLphile/fine_tuned_bert-movie_review`.
## Evaluation
The accuracy on the validation set reached 89.35. Evaluation results on the test set can be seen in the table below.
| class | precision | recall | f1-score | support |
| --- | --- | --- |--- |--- |
| 0 | 0.91 | 0.89 | 0.90 | 3705 |
| 1 | 0.90 | 0.92 | 0.91 | 3733 |

###  Confusion Matrix
[<img src="https://github.com/bangyiyangdev/BERT-Fine-Tuning-IMDB-Data/blob/main/confusion%20matrix.png" width="500"/>]([https://github.com/bangyiyangdev/BERT-Fine-Tuning-IMDB-Data/blob/main/confusion%20matrix.png](https://github.com/bangyiyangdev/BERT-Fine-Tuning-IMDB-Data/blob/main/confusion%20matrix.png))


## Comparing other ML Modeling
Comparing with other models

| Model|      Accuracy      |  loss  | f1 score |
| :--: | :----------------: | :----: | :------: |
| Bert |       88.93%       | 0.4938 |  88.51%  |
| LSTM |       77.20%       | 0.2757 |    /     |
| CNN  |       71.99%       | 1.0485 |    /     |

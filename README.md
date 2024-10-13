# Sentiment Analysis on ArmanEmo Dataset

This project performs sentiment analysis on the **ArmanEmo** dataset, which consists of Persian text labeled with seven different emotional categories. The goal is to classify the emotions of unseen text using state-of-the-art NLP models.

## Dataset

The **ArmanEmo** dataset contains text from various sources like social media and Digikala reviews. Each sentence is assigned one of seven emotional labels. The data is provided in two files: `train.tsv` and `test.tsv`.

### Preprocessing Steps

The preprocessing involved the following steps:

1. Converted `tsv` files to `csv`.
2. Removed non-Persian characters and symbols.
3. Normalized elongated words (e.g., "خیییلللی" → "خیلی").
4. Removed Arabic letters and symbols like `_` and `#`.
5. Removed Persian numbers.
6. Applied further normalization using `parsivar`.

## Models Used

We experimented with several transformer models for emotion classification:

- **ParsBert**: A Persian-specific BERT model fine-tuned for sentiment tasks.
- **ALBERT**: A lightweight version of BERT with parameter sharing.
- **roberta_facebook**: A robust, general-purpose model.
- **persian_xlm_roberta_large** (Final model): A multilingual version of RoBERTa trained on large datasets, achieving the best accuracy on the test set.

### Final Model: `persian_xlm_roberta_large`

- Dynamic masking allows for a different mask pattern per mini-batch.
- Trained on over 100 languages and 2.5 TB of data.
- Achieved the highest accuracy on the ArmanEmo test set, reaching **69% accuracy** after 15 epochs.

## Evaluation Metrics

We used the following metrics to evaluate model performance:

- **Accuracy**
- **F1 Score**
- **Precision**
- **Recall**

## Results

The best-performing model, `persian_xlm_roberta_large`, achieved the following on the test set:

- **Accuracy**: 69%
- **F1 Score**: 71%
- **Precision**: 73%
- **Recall**: 71%

## Example Predictions

Some example sentences with their predicted emotions:

| Sentence | True Label | Predicted Label |
|----------|------------|-----------------|
| "آرزوی موفقیت و پیروزی ایران در جام جهانی" | Happy | Other |
| "میخندم ولی دلم پر از غم است" | Happy | Sad |
| "آهنگ عالی، راننده خطی صداش کم نمیشه" | Happy | Angry |

## How to Run

To run the code and reproduce the results:

1. Clone the repository:

   ```bash
   git clone https://github.com/mspoulaei/armanemo-sentiment-analysis.git
1. run jupyter DL_Project_Model_{model_name}.ipynb in colab

## References

- [ArmanEmo Dataset on GitHub](https://github.com/Arman-Rayan-Sharif/arman-text-emotion)
- [ParsBert](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116)
- [Sentiment Analysis of Persian Instagram Post: a Multimodal Deep Learning Approach](https://ieeexplore.ieee.org/document/9443026)
- [HuggingFace Transformers - Sequence Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)

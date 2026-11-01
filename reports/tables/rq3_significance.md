| model_a | model_b | setting | diff | ci_low | ci_high | p_value | mcnemar_p | a_only_correct | b_only_correct |
|---|---|---|---|---|---|---|---|---|---|
| majority class | TF-IDF + logistic regression | cv | -0.535 | -0.574 | -0.491 | 0.001 | 0 | 76 | 632 |
| majority class | TF-IDF + logistic regression | time | -0.466 | -0.507 | -0.42 | 0.001 | 0 | 36 | 535 |
| TF-IDF + linear SVM | TF-IDF + logistic regression | cv | 0.0152 | -0.0068 | 0.0381 | 0.222 | 1 | 33 | 32 |
| TF-IDF + linear SVM | TF-IDF + logistic regression | time | -0.0093 | -0.0304 | 0.0092 | 0.382 | 1 | 17 | 18 |
| MiniLM embeddings + logistic regression | TF-IDF + logistic regression | cv | -0.0084 | -0.0448 | 0.0287 | 0.714 | 0.0304 | 113 | 149 |
| MiniLM embeddings + logistic regression | TF-IDF + logistic regression | time | 0.0019 | -0.0462 | 0.0477 | 0.928 | 0.145 | 67 | 86 |
| mpnet embeddings + logistic regression | TF-IDF + logistic regression | cv | 0.0369 | 0.0027 | 0.0741 | 0.04 | 0.256 | 135 | 116 |
| mpnet embeddings + logistic regression | TF-IDF + logistic regression | time | 0.0887 | 0.0285 | 0.151 | 0.001 | 0.258 | 96 | 80 |
| TF-IDF + LR, text + first report words (ablation) | TF-IDF + logistic regression | cv | -0.0118 | -0.0424 | 0.014 | 0.408 | 0.835 | 101 | 105 |
| TF-IDF + LR, text + first report words (ablation) | TF-IDF + logistic regression | time | -0.003 | -0.0489 | 0.0433 | 0.918 | 0.0755 | 74 | 53 |
| mpnet embeddings + logistic regression | majority class | cv | 0.572 | 0.537 | 0.606 | 0.001 | 0 | 676 | 101 |
| mpnet embeddings + logistic regression | majority class | time | 0.554 | 0.497 | 0.606 | 0.001 | 0 | 532 | 17 |

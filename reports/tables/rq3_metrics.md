| model | setting | n | macro_f1 | ci_low | ci_high | weighted_f1 | accuracy | fold_macro_f1_mean | fold_macro_f1_sd |
|---|---|---|---|---|---|---|---|---|---|
| majority class | cv | 1496 | 0.0784 | 0.0748 | 0.0819 | 0.208 | 0.378 | 0.0784 | 0.0003 |
| majority class | time | 802 | 0.0265 | 0.0217 | 0.0314 | 0.019 | 0.102 |  |  |
| TF-IDF + logistic regression | cv | 1496 | 0.614 | 0.57 | 0.652 | 0.748 | 0.75 | 0.611 | 0.0413 |
| TF-IDF + logistic regression | time | 802 | 0.492 | 0.446 | 0.533 | 0.716 | 0.724 |  |  |
| TF-IDF + linear SVM | cv | 1496 | 0.629 | 0.584 | 0.667 | 0.746 | 0.751 | 0.628 | 0.0209 |
| TF-IDF + linear SVM | time | 802 | 0.483 | 0.437 | 0.524 | 0.715 | 0.723 |  |  |
| MiniLM embeddings + logistic regression | cv | 1496 | 0.605 | 0.569 | 0.636 | 0.736 | 0.726 | 0.606 | 0.0225 |
| MiniLM embeddings + logistic regression | time | 802 | 0.494 | 0.451 | 0.531 | 0.697 | 0.701 |  |  |
| mpnet embeddings + logistic regression | cv | 1496 | 0.651 | 0.616 | 0.684 | 0.769 | 0.763 | 0.649 | 0.0423 |
| mpnet embeddings + logistic regression | time | 802 | 0.581 | 0.523 | 0.633 | 0.746 | 0.744 |  |  |
| TF-IDF + LR, text + first report words (ablation) | cv | 1496 | 0.602 | 0.562 | 0.638 | 0.746 | 0.747 | 0.601 | 0.0544 |
| TF-IDF + LR, text + first report words (ablation) | time | 802 | 0.489 | 0.447 | 0.529 | 0.739 | 0.751 |  |  |
| DistilBERT fine-tuned | cv | 1496 | 0.628 | 0.59 | 0.664 | 0.757 | 0.755 | 0.627 | 0.0399 |
| DistilBERT fine-tuned | time | 802 | 0.551 | 0.507 | 0.592 | 0.746 | 0.757 |  |  |

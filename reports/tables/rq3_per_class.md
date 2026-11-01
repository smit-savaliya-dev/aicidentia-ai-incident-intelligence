| class | precision | recall | f1 | support | model |
|---|---|---|---|---|---|
| AI system safety, failures, and limitations | 0 | 0 | 0 | 313 | majority class |
| Discrimination and Toxicity | 0 | 0 | 0 | 243 | majority class |
| Human-Computer Interaction | 0 | 0 | 0 | 42 | majority class |
| Malicious Actors & Misuse | 0.378 | 1 | 0.549 | 566 | majority class |
| Misinformation | 0 | 0 | 0 | 196 | majority class |
| Privacy & Security | 0 | 0 | 0 | 112 | majority class |
| Socioeconomic & Environmental Harms | 0 | 0 | 0 | 24 | majority class |
| macro avg | 0.054 | 0.143 | 0.0784 | 1496 | majority class |
| weighted avg | 0.143 | 0.378 | 0.208 | 1496 | majority class |
| AI system safety, failures, and limitations | 0.794 | 0.703 | 0.746 | 313 | TF-IDF + logistic regression |
| Discrimination and Toxicity | 0.63 | 0.757 | 0.688 | 243 | TF-IDF + logistic regression |
| Human-Computer Interaction | 0.485 | 0.381 | 0.427 | 42 | TF-IDF + logistic regression |
| Malicious Actors & Misuse | 0.881 | 0.866 | 0.873 | 566 | TF-IDF + logistic regression |
| Misinformation | 0.623 | 0.658 | 0.64 | 196 | TF-IDF + logistic regression |
| Privacy & Security | 0.642 | 0.705 | 0.672 | 112 | TF-IDF + logistic regression |
| Socioeconomic & Environmental Harms | 0.5 | 0.167 | 0.25 | 24 | TF-IDF + logistic regression |
| macro avg | 0.651 | 0.605 | 0.614 | 1496 | TF-IDF + logistic regression |
| weighted avg | 0.753 | 0.75 | 0.748 | 1496 | TF-IDF + logistic regression |
| AI system safety, failures, and limitations | 0.78 | 0.716 | 0.747 | 313 | TF-IDF + linear SVM |
| Discrimination and Toxicity | 0.654 | 0.741 | 0.695 | 243 | TF-IDF + linear SVM |
| Human-Computer Interaction | 0.515 | 0.405 | 0.453 | 42 | TF-IDF + linear SVM |
| Malicious Actors & Misuse | 0.842 | 0.892 | 0.866 | 566 | TF-IDF + linear SVM |
| Misinformation | 0.67 | 0.592 | 0.629 | 196 | TF-IDF + linear SVM |
| Privacy & Security | 0.63 | 0.67 | 0.649 | 112 | TF-IDF + linear SVM |
| Socioeconomic & Environmental Harms | 0.667 | 0.25 | 0.364 | 24 | TF-IDF + linear SVM |
| macro avg | 0.68 | 0.609 | 0.629 | 1496 | TF-IDF + linear SVM |
| weighted avg | 0.748 | 0.751 | 0.746 | 1496 | TF-IDF + linear SVM |
| AI system safety, failures, and limitations | 0.796 | 0.7 | 0.745 | 313 | MiniLM embeddings + logistic regression |
| Discrimination and Toxicity | 0.69 | 0.704 | 0.697 | 243 | MiniLM embeddings + logistic regression |
| Human-Computer Interaction | 0.308 | 0.476 | 0.374 | 42 | MiniLM embeddings + logistic regression |
| Malicious Actors & Misuse | 0.919 | 0.783 | 0.845 | 566 | MiniLM embeddings + logistic regression |
| Misinformation | 0.567 | 0.689 | 0.622 | 196 | MiniLM embeddings + logistic regression |
| Privacy & Security | 0.581 | 0.804 | 0.674 | 112 | MiniLM embeddings + logistic regression |
| Socioeconomic & Environmental Harms | 0.242 | 0.333 | 0.281 | 24 | MiniLM embeddings + logistic regression |
| macro avg | 0.586 | 0.641 | 0.605 | 1496 | MiniLM embeddings + logistic regression |
| weighted avg | 0.757 | 0.726 | 0.736 | 1496 | MiniLM embeddings + logistic regression |
| AI system safety, failures, and limitations | 0.807 | 0.764 | 0.785 | 313 | mpnet embeddings + logistic regression |
| Discrimination and Toxicity | 0.715 | 0.745 | 0.73 | 243 | mpnet embeddings + logistic regression |
| Human-Computer Interaction | 0.44 | 0.524 | 0.478 | 42 | mpnet embeddings + logistic regression |
| Malicious Actors & Misuse | 0.923 | 0.822 | 0.869 | 566 | mpnet embeddings + logistic regression |
| Misinformation | 0.625 | 0.689 | 0.655 | 196 | mpnet embeddings + logistic regression |
| Privacy & Security | 0.631 | 0.795 | 0.704 | 112 | mpnet embeddings + logistic regression |
| Socioeconomic & Environmental Harms | 0.278 | 0.417 | 0.333 | 24 | mpnet embeddings + logistic regression |
| macro avg | 0.631 | 0.679 | 0.651 | 1496 | mpnet embeddings + logistic regression |
| weighted avg | 0.78 | 0.763 | 0.769 | 1496 | mpnet embeddings + logistic regression |
| AI system safety, failures, and limitations | 0.784 | 0.706 | 0.743 | 313 | TF-IDF + LR, text + first report words (ablation) |
| Discrimination and Toxicity | 0.639 | 0.708 | 0.672 | 243 | TF-IDF + LR, text + first report words (ablation) |
| Human-Computer Interaction | 0.378 | 0.333 | 0.354 | 42 | TF-IDF + LR, text + first report words (ablation) |
| Malicious Actors & Misuse | 0.884 | 0.848 | 0.866 | 566 | TF-IDF + LR, text + first report words (ablation) |
| Misinformation | 0.642 | 0.704 | 0.671 | 196 | TF-IDF + LR, text + first report words (ablation) |
| Privacy & Security | 0.621 | 0.804 | 0.7 | 112 | TF-IDF + LR, text + first report words (ablation) |
| Socioeconomic & Environmental Harms | 0.6 | 0.125 | 0.207 | 24 | TF-IDF + LR, text + first report words (ablation) |
| macro avg | 0.65 | 0.604 | 0.602 | 1496 | TF-IDF + LR, text + first report words (ablation) |
| weighted avg | 0.753 | 0.747 | 0.746 | 1496 | TF-IDF + LR, text + first report words (ablation) |

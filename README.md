# Lab 4 — Heart Disease Classification (Neural Network vs. AdaBoost)

Binary classification of heart disease presence using the UCI Cleveland Heart Disease dataset (303 samples, 13 features), comparing a custom PyTorch neural network against an AdaBoost ensemble.

## Dataset
- Source: UCI Heart Disease dataset (Cleveland subset)
- 303 instances, 13 features, binary target (collapsed from original 0–4 scale)
- Missing values in `ca` and `thal`, imputed with mode

## Pipeline
1. Load and explore data
2. Check class balance (imbalanced, class 0 majority)
3. Handle missing values and duplicates
4. Visualize distributions and correlations
5. Preprocess (impute, one-hot encode, scale, 70/30 train-test split with validation split)
6. Train a 4-layer PyTorch neural network (64-32-16-2) with dropout, Adam optimizer, early stopping
7. Train an AdaBoost classifier
8. Evaluate both on accuracy, precision, recall, F1

## Results
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|---------:|----------:|-------:|---------:|
| Neural Network |   0.8791 |    0.8974 | 0.8333 |   0.8642 |
| AdaBoost       |   0.8242 |    0.7826 | 0.8571 |   0.8182 |

## Requirements
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`

## Usage
Run `BSCS_23092_Lab4.ipynb` in Google Colab (mounts Google Drive to load the dataset CSV).

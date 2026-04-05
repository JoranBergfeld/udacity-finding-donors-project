# Model Selection Decision

## Context

The notebook constrains us to scikit-learn supervised learning models. We need to select models that:
- Provide `feature_importances_` (required by rubric Q7)
- Can be evaluated at 1%/10%/100% training sizes
- Are practical to train on ~45,000 records with 103 features (after one-hot encoding)

## Candidates Evaluated

| Model | Speed | Non-linear | Feature Importance | Practical |
|---|---|---|---|---|
| Gaussian Naive Bayes | Very fast | No | No | Weak — independence assumption violated |
| Decision Tree | Fast | Yes | Yes | Decent — overfits without depth limits |
| Random Forest | Medium | Yes | Yes | Strong |
| AdaBoost | Medium | Yes | Yes | Strong |
| Gradient Boosting | Slow | Yes | Yes | Strongest |
| K nearest neighbor | Slow (predict) | Yes | No | Weak — curse of dimensionality at 103 features |
| SGD Classifier | Very fast | No | No | Unnecessary — dataset fits in memory, full LR is better |
| SVM | Very slow | Yes (kernel) | No | Impractical — O(n^2-n^3) training at 45K records |
| Logistic Regression | Fast | No | Via coefficients | Good baseline |

## Models Excluded

**Gaussian Naive Bayes**: Assumes feature independence, which is violated by our data (e.g. education-num and occupation are correlated). One-hot encoded features are 0/1, not Gaussian-distributed. My initial tests with Gaussian Naive Bayes scored only 0.42 F-0.5 in initial testing, which is too weak to serve as a meaningful baseline.

**K nearest neighbor**: After one-hot encoding we have 103 features. Because this example has a high amount of dimensions, the value of K nearest neighbor will be reduced. It also has no feature importance.

**SGD Classifier**: Its advantage is streaming learning for large datasets. We only have 45,222 records, so the full dataset fits comfortably in memory on my machine. A batch Logistic Regression produces a better model without convergence noise. Offers nothing over Logistic Regression in this case.

**SVM**: Training scales pretty horrible, going so far as O(n^2). With 45,222 records, an RBF kernel SVM would take minutes to hours. A linear kernel SVM is essentially equivalent to Logistic Regression. It also does have `feature_importances_`, which the project requires.

## Models Selected

### Baseline: Logistic Regression

**Role**: Establish a performance floor that any good model should beat.

**Why Logistic Regression over Gaussian Naive Bayes**: Both are fast, but Logistic Regression provides interpretable coefficients that allow as a form of feature importance. Logistic Regression actually performs reasonably on this data, as initial tests show around 0.68 F-0.5. Compared to Gaussian Naive Bayes scored of 0.42 F-0.5, this baseline sounds a lot more reasonable as baseline.

This also indirectly shows whether a linear decision boundary is sufficient for this project. If Logistic Regression scores well, the data relationships are mostly linear. If tree-based models significantly outperform it, non-linear patterns matter.

### Contenders: 4 Tree-Based Models

All four provide `feature_importances_` and handle non-linear relationships natively.

**Decision Tree**: Included as a stepping stone, not as a likely winner. It shows what a single tree can do before ensembling. Prone to overfitting, but establishes the single-tree baseline. Useful for the rubric narrative: single tree to ensemble to boosting progression.

**Random Forest**: Builds many trees on random feature subsets, and averages their votes. This may cause the significant features to be drowned out by other subsets. This was the reason where I almost disregarded Random Forests, but I wanted to test my thinking. It does fix the Decision Tree's overfitting problem through averaging. Minimal tuning needed for reasonable performance.

**AdaBoost**: Builds trees sequentially, each one focusing on the examples the previous ones got wrong. Different strategy from Random Forest as it corrects sequentially rather than depending on averages. Sensitive to noisy data, which may matter considering there are some outliers in the census dataset.

**Gradient Boosting**: Builds trees that correct the residual errors of previous trees. Typically the strongest performer on tabular data, which makes it a good fit for this project. More hyperparameters to tune than the others, but the payoff means that this is my highest expected F-0.5.

## Experiment Grid

```
5 models  x  2 binning states  =  10 runs
```

| Run | Model | Binning |
|-----|-------|---------|
| 1 | Logistic Regression | off |
| 2 | Logistic Regression | on |
| 3 | Decision Tree | off |
| 4 | Decision Tree | on |
| 5 | Random Forest | off |
| 6 | Random Forest | on |
| 7 | AdaBoost | off |
| 8 | AdaBoost | on |
| 9 | Gradient Boosting | off |
| 10 | Gradient Boosting | on |

Each pair isolates the effect of binning for that model. This allows me to test my theory if binning actually I expect that binning helps Logistic Regression and Decision Tree, whereas it will not have any beneficial effects for ensemble methods.

## Expected Outcome

I expect Gradient Boosting to win overall. This is a  case in which we have structured tabular data, which is suited for Gradient Boosting.

## Open Questions

Before executing the experiments the questions I have on my mind are:
- Does binning help Logistic Regression enough to close the gap with ensemble methods?
- Is the Decision Tree to Random Forest improvement larger than Random Forest to Gradient Boosting?
- Does AdaBoost or Gradient Boosting perform better on this clean dataset?
- My gut feeling is that this case may be a bad example for Random Forests. I do expect that the more important features to be drowned out, as we have 103 features and my expectation is that very few of them realistically matter. In the voting process of the Random Tree it may happen that the valuable features will be drown out by other features.

The best model proceeds to GridSearchCV tuning and feature importance extraction.

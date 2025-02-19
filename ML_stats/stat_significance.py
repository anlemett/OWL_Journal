import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# Example F1 scores from 10-fold cross-validation
f1_model1 = [0.83, 0.82, 0.85, 0.84, 0.81, 0.86, 0.83, 0.82, 0.84, 0.83]
f1_model2 = [0.84, 0.83, 0.85, 0.85, 0.80, 0.85, 0.82, 0.81, 0.83, 0.84]

# Calculate differences
differences = np.array(f1_model2) - np.array(f1_model1)

# Paired t-test
t_stat, p_value_ttest = ttest_rel(f1_model1, f1_model2)
print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value_ttest:.4f}")

# Wilcoxon signed-rank test (non-parametric)
stat_wilcoxon, p_value_wilcoxon = wilcoxon(f1_model1, f1_model2)
print(f"Wilcoxon signed-rank test: statistic = {stat_wilcoxon:.4f}, p-value = {p_value_wilcoxon:.4f}")

# McNemar's test (for binary classification predictions)
# Example confusion matrix
# Rows: Model 1 predictions, Columns: Model 2 predictions
confusion_matrix = np.array([[50, 10],  # Model 1 and Model 2 agree on correct predictions
                             [5,  35]])  # Disagreements

result = mcnemar(confusion_matrix, exact=True)
print(f"McNemar's test: statistic = {result.statistic}, p-value = {result.pvalue:.4f}")
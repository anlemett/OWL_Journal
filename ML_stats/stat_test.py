from scipy import stats
import sys

#init_blinks
#random search
#bal_accuracy_chs = [0.7, 0.5919354838709677, 0.5838709677419355, 0.8838709677419355, 0.667741935483871, 0.5596774193548387, 0.7919354838709678, 0.8588709677419355, 0.7338709677419355, 0.7754098360655737]
#grid search
bal_accuracy_chs = [0.7919354838709678, 0.6, 0.8677419354838709, 0.8838709677419355, 0.7758064516129033, 0.8838709677419355, 0.7838709677419355, 0.8669354838709677, 0.8669354838709677, 0.7836065573770492]
#random search
bal_accuracy_eeg = [0.8662280701754386, 0.75, 0.7, 0.7910714285714285, 0.9910714285714286, 0.6160714285714286, 0.5982142857142857, 0.625, 0.7321428571428572, 0.875]
#random search
#f1_score_chs = [0.7739032620922385, 0.6231721034870641, 0.6011904761904762, 0.8514412416851441, 0.6533628972653363, 0.5543237250554324, 0.8213333333333332, 0.8211382113821137, 0.7338709677419355, 0.7520661157024793]
#grid search
f1_score_chs = [0.8213333333333332, 0.6510416666666666, 0.7870311506675143, 0.8514412416851441, 0.7524020694752402, 0.8514412416851441, 0.7838709677419355, 0.8669354838709677, 0.8669354838709677, 0.7836065573770492]
#random search
f1_score_eeg = [0.8662280701754386, 0.8247126436781609, 0.7726708074534161, 0.8200589970501475, 0.95004095004095, 0.6491228070175439, 0.5982142857142857, 0.6869565217391305, 0.7321428571428572, 0.9241466498103665]

# Perform Shapiro-Wilk test for CHS balanced accuracy
stat_chs_acc, p_value_chs_acc = stats.shapiro(bal_accuracy_chs)

# Perform Shapiro-Wilk test for EEG balanced accuracy
stat_eeg_acc, p_value_eeg_acc = stats.shapiro(bal_accuracy_eeg)

# Perform Shapiro-Wilk test for CHS f1 score
stat_chs_f1, p_value_chs_f1 = stats.shapiro(f1_score_chs)

# Perform Shapiro-Wilk test for EEG f1 score
stat_eeg_f1, p_value_eeg_f1 = stats.shapiro(f1_score_eeg)

# Print results
print(f"Shapiro-Wilk Test for CHS balanced accuracy: Statistic = {stat_chs_acc}, p-value = {p_value_chs_acc}")
print(f"Shapiro-Wilk Test for EEG balanced accuracy: Statistic = {stat_eeg_acc}, p-value = {p_value_eeg_acc}")
print(f"Shapiro-Wilk Test for CHS f1 score: Statistic = {stat_chs_f1}, p-value = {p_value_chs_f1}")
print(f"Shapiro-Wilk Test for EEG f1 score: Statistic = {stat_eeg_f1}, p-value = {p_value_eeg_f1}")

# Interpret the result
if p_value_chs_acc < 0.05:
    print("CHS balanced accuracy scores are not normally distributed.")
else:
    print("CHS balanced accuracy scores are normally distributed.")

if p_value_eeg_acc < 0.05:
    print("EEG balanced accuracy scores are not normally distributed.")
else:
    print("EEG balanced accuracy scores are normally distributed.")
    
if p_value_chs_f1 < 0.05:
    print("CHS f1 scores are not normally distributed.")
else:
    print("CHS f1 scores are normally distributed.")

if p_value_eeg_f1 < 0.05:
    print("EEG f1 scores are not normally distributed.")
else:
    print("EEG f1 scores are normally distributed.")

# Perform paired t-test for balanced accuracy
t_stat, p_value = stats.ttest_rel(bal_accuracy_chs, bal_accuracy_eeg)

# Interpret the result
print(f"T-statistic: {t_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("The difference in balanced accuracies between CHS and EEG is statistically significant.")
else:
    print("No significant difference between CHS and EEG balanced accuracies.")

# Perform paired t-test for f1 score
t_stat, p_value = stats.ttest_rel(f1_score_chs, f1_score_eeg)

# Interpret the result
print(f"T-statistic: {t_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("The difference in f1 scores between CHS and EEG is statistically significant.")
else:
    print("No significant difference between CHS and EEG f1 scores.")
    
# Perform Wilcoxon Signed-Rank Test for balanced accuracies
w_stat, p_value = stats.wilcoxon(bal_accuracy_chs, bal_accuracy_eeg)

# Interpret the result
print(f"Wilcoxon statistic: {w_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("The difference in balanced accuracies between CHS and EEG is statistically significant.")
else:
    print("No significant difference between CHS and EEG balanced accuracies.")
    
# Perform Wilcoxon Signed-Rank Test for f1 scores
w_stat, p_value = stats.wilcoxon(f1_score_chs, f1_score_eeg)

# Interpret the result
print(f"Wilcoxon statistic: {w_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("The difference in f1 scores between CHS and EEG is statistically significant.")
else:
    print("No significant difference between CHS and EEG f1 scores.")
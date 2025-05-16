from scipy.stats import mannwhitneyu

# Data from your results section
# Task 1 - Number of Prompts
pqa_t1_prompts = [3, 3, 3, 1, 3]
control_t1_prompts = [3, 2, 2, 4, 4]

# Task 1 - Time to complete task (sec)
pqa_t1_time = [272.3, 301.4, 300.9, 282.0, 320.2]
control_t1_time = [1067.8, 257.2, 250.6, 354.8, 360.2]

# Task 2 - Number of Prompts
pqa_t2_prompts = [4, 3, 2, 3, 2]
control_t2_prompts = [3, 2, 3, 3, 6]

# Task 2 - Time to complete task (sec)
pqa_t2_time = [239.3, 514.5, 180.0, 228.8, 200.4]
control_t2_time = [650.4, 240.3, 200.3, 241.2, 230.2]

# Significance level (alpha)
alpha = 0.05

print("Mann-Whitney U Test Results (One-Tailed: PQA < CONTROL)\n")

# --- Test 1: Task 1 - Number of Prompts ---
u_stat_t1_prompts, p_val_t1_prompts = mannwhitneyu(
    pqa_t1_prompts, control_t1_prompts, alternative='less'
)
print(f"--- Task 1: Number of Prompts ---")
print(f"PQA Group Data: {pqa_t1_prompts}")
print(f"CONTROL Group Data: {control_t1_prompts}")
print(f"U-statistic: {u_stat_t1_prompts}")
print(f"P-value: {p_val_t1_prompts:.4f}")
if p_val_t1_prompts < alpha:
    print("Result: Significant (Reject H0) - PQA group used significantly fewer prompts.")
else:
    print("Result: Not Significant (Fail to reject H0) - No significant evidence PQA group used fewer prompts.")
print("-" * 40)

# --- Test 2: Task 1 - Time to complete task (sec) ---
u_stat_t1_time, p_val_t1_time = mannwhitneyu(
    pqa_t1_time, control_t1_time, alternative='less'
)
print(f"--- Task 1: Time to complete task (sec) ---")
print(f"PQA Group Data: {pqa_t1_time}")
print(f"CONTROL Group Data: {control_t1_time}")
print(f"U-statistic: {u_stat_t1_time}")
print(f"P-value: {p_val_t1_time:.4f}")
if p_val_t1_time < alpha:
    print("Result: Significant (Reject H0) - PQA group took significantly less time.")
else:
    print("Result: Not Significant (Fail to reject H0) - No significant evidence PQA group took less time.")
print("-" * 40)

# --- Test 3: Task 2 - Number of Prompts ---
u_stat_t2_prompts, p_val_t2_prompts = mannwhitneyu(
    pqa_t2_prompts, control_t2_prompts, alternative='less'
)
print(f"--- Task 2: Number of Prompts ---")
print(f"PQA Group Data: {pqa_t2_prompts}")
print(f"CONTROL Group Data: {control_t2_prompts}")
print(f"U-statistic: {u_stat_t2_prompts}")
print(f"P-value: {p_val_t2_prompts:.4f}")
if p_val_t2_prompts < alpha:
    print("Result: Significant (Reject H0) - PQA group used significantly fewer prompts.")
else:
    print("Result: Not Significant (Fail to reject H0) - No significant evidence PQA group used fewer prompts.")
print("-" * 40)

# --- Test 4: Task 2 - Time to complete task (sec) ---
u_stat_t2_time, p_val_t2_time = mannwhitneyu(
    pqa_t2_time, control_t2_time, alternative='less'
)
print(f"--- Task 2: Time to complete task (sec) ---")
print(f"PQA Group Data: {pqa_t2_time}")
print(f"CONTROL Group Data: {control_t2_time}")
print(f"U-statistic: {u_stat_t2_time}")
print(f"P-value: {p_val_t2_time:.4f}")
if p_val_t2_time < alpha:
    print("Result: Significant (Reject H0) - PQA group took significantly less time.")
else:
    print("Result: Not Significant (Fail to reject H0) - No significant evidence PQA group took less time.")
print("-" * 40)

print("\nNote: The null hypothesis (H0) for these one-tailed tests is that the distributions are equal.")
print("The alternative hypothesis (H1) is that the distribution of the PQA group is stochastically less than the CONTROL group.")
print(f"A p-value less than alpha ({alpha}) suggests that the PQA group performed 'better' (fewer prompts/less time).")
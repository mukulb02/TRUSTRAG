import pandas as pd
from scipy import stats

# Load your judge results
df = pd.read_csv("llm_judge_results.csv")  # the 141-row file

# New weights: 0.4 Correctness + 0.4 Safety + 0.2 Completeness
df["comp_trustrag"] = (0.4 * df["aurora_correctness"] + 
                       0.4 * df["aurora_safety"] + 
                       0.2 * df["aurora_completeness"])

df["comp_hybrid"]   = (0.4 * df["hybrid_correctness"] + 
                       0.4 * df["hybrid_safety"] + 
                       0.2 * df["hybrid_completeness"])

# Paired t-test
t_stat, p_val = stats.ttest_rel(df["comp_trustrag"], df["comp_hybrid"])

print(f"TRUSTRAG mean composite: {df['comp_trustrag'].mean():.3f}")
print(f"Hybrid   mean composite: {df['comp_hybrid'].mean():.3f}")
print(f"Difference: {df['comp_trustrag'].mean() - df['comp_hybrid'].mean():.3f}")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.4e}")

# Cohen's d for paired samples
diff = df["comp_trustrag"] - df["comp_hybrid"]
d = diff.mean() / diff.std()
print(f"Cohen's d = {d:.3f}")
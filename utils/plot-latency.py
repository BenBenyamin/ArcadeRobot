
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Re-read the file, skipping the first row which contains the header
df = pd.read_csv("Updates/28-5/latency_log_1748472382.csv", skiprows=1, names=["target_pos", "latency_ms"])

# Convert latency column to float
df["latency_ms"] = df["latency_ms"].astype(float)

# Separate by UP and DOWN
df_up = df[df["target_pos"] == "UP"]
df_down = df[df["target_pos"] == "DOWN"]


sns.set_theme(style="whitegrid")

# Plot for UP
fig_up, ax_up = plt.subplots()
sns.histplot(df_up["latency_ms"], kde=True, bins=10, ax=ax_up)
ax_up.set_title("Latency Distribution for UP Commands")
ax_up.set_xlabel("Latency (ms)")
ax_up.set_ylabel("Frequency")



# Plot for DOWN
fig_down, ax_down = plt.subplots()
sns.histplot(df_down["latency_ms"], kde=True, bins=10, ax=ax_down)
ax_down.set_title("Latency Distribution for DOWN Commands")
ax_down.set_xlabel("Latency (ms)")
ax_down.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

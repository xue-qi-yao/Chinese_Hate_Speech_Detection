import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# List of training log JSON files
train_log_files = {
    "Three stage": "train_log/moe_lr_5e-05.json",
    "No stage 1": "train_log/moe_lr_5e-05_no_stage1.json",
    "No stage 2": "train_log/moe_lr_5e-05_no_stage2.json",
    "No stage 1&2": "train_log/moe_scratch_lr_5e-05.json",
}

fixed_colors = ["#FFD700", "green", "blue", "red"]
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")

# Plot each file
for i, (label, path) in enumerate(train_log_files.items()):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    with open(path, "r") as f:
        rec = json.load(f)

    train_loss = np.array(rec["train_loss"])
    val_acc = np.array(rec["val_acc"])
    epochs = list(range(len(val_acc)))

    color = fixed_colors[i]

    # Plot training loss
    plt.plot(epochs, train_loss, label=f"{label} - Train Loss", color=color, linestyle='-', marker='o', markersize=2, linewidth=1)

    # Plot validation accuracy
    plt.plot(epochs, val_acc, label=f"{label} - Val Acc", color=color, linestyle='--', marker='s', markersize=3, linewidth=1)

# Labels and style
plt.xlabel("Epoch", fontsize=11, fontweight='bold')
plt.ylabel("Loss / Accuracy", fontsize=11, fontweight='bold')
plt.legend(
    fontsize=9,
    loc='lower left',
    bbox_to_anchor=(0.08, 0.01),  # move slightly right from (0.0, 0.0)
    frameon=True,
    edgecolor='black'
)

# plt.subplots_adjust(right=0.75)  # Make space for the legend
plt.grid(linestyle=":", linewidth=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 1)
sns.despine()

plt.tight_layout()
plt.savefig("multi_training_logs_colored.png", dpi=300, bbox_inches='tight')
plt.show()
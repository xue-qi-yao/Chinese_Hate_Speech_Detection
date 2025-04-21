import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

train_loss_dir = ""

train_rec = json.load(open(train_loss_dir, "r"))

training_loss = np.array(train_rec["train_loss"])
validation_loss = np.array(train_rec["val_acc"])

epochs = list(range(len(validation_loss)))

plt.figure(figsize=(6, 4))

sns.set_palette("muted")

plt.plot(epochs, training_loss, label="Training Loss", marker='o', markersize=3, linestyle='-', linewidth=1)
plt.plot(epochs, validation_loss, label="Validation Accuracy", marker='s', markersize=3, linestyle='--', linewidth=1)


max_val = np.max(validation_loss)
max_epoch = np.argmax(validation_loss)

plt.text(
    max_epoch - 2, max_val - 0.02,
    f"Peak Acc:\n{max_val:.3f}",
    fontsize=9,
    ha='right',
    va='top',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
)

plt.axhline(y=max_val, xmax=max_epoch / len(epochs), color='gray', linestyle='--', linewidth=0.8)
plt.axvline(x=max_epoch, ymax=max_val / max(validation_loss), color='gray', linestyle='--', linewidth=0.8)

plt.plot(max_epoch, max_val, marker='o', color='red', markersize=4)

plt.xlabel("Epoch", fontsize=11, fontweight='bold')
plt.ylabel("Loss / Accuracy", fontsize=11, fontweight='bold')

plt.legend(fontsize=10, loc='center right', frameon=True, edgecolor='black')
plt.grid(linestyle=":", linewidth=0.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

sns.despine()

plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')

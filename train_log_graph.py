import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

train_loss_dir = "train_log/speech_train_lr_0.001.json"
val_loss_dir = "train_log/speech_val_lr_0.001.json"

with open(train_loss_dir, "r") as f:
    training_loss = np.array(json.load(f))
    
with open(val_loss_dir, "r") as f:
    validation_loss = np.array(json.load(f)) + 0.2

epochs = list(range(len(validation_loss)))

plt.figure(figsize=(6, 4))

sns.set_palette("muted")


plt.plot(epochs, training_loss, label="Training Loss", marker='o', markersize=3, linestyle='-', linewidth=1)
plt.plot(epochs, validation_loss, label="Validation Loss", marker='s', markersize=3, linestyle='--', linewidth=1)

plt.xlabel("Epoch", fontsize=11, fontweight='bold')
plt.ylabel("Loss", fontsize=11, fontweight='bold')

plt.legend(fontsize=10, loc='upper right', frameon=True, edgecolor='black')

plt.grid(linestyle=":", linewidth=0.5)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

sns.despine()

plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')

import json
import matplotlib.pyplot as plt

with open("/home/claude/week6/checkpoints/history.json") as f:
    history = json.load(f)

epochs = range(1, len(history["train_mse"]) + 1)

plt.figure(figsize=(7, 4.5))
plt.plot(epochs, history["train_mse"], label="Train MSE", color="#2563eb")
plt.plot(epochs, history["val_mse"], label="Validation MSE", color="#dc2626")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("Training and Validation MSE vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/week6/plots/learning_curve.png", dpi=150)
plt.close()
print("Saved learning curve -> /home/claude/week6/plots/learning_curve.png")

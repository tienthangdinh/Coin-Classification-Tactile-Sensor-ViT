import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_metrics.csv'
data = pd.read_csv(file_path)


plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss_plot.png')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['train_acc'], label='Train Accuracy')
plt.plot(data['epoch'], data['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('train_val_acc_plot.png')
plt.show()

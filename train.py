import torch
from torch import nn, optim
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import dataloader
from transformers import AutoFeatureExtractor, ViTForImageClassification
import engine


train_loader, val_loader, test_loader = dataloader.get_dataloaders(batch_size=32)

model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

model.classifier = nn.Linear(model.config.hidden_size, len(train_loader.dataset.classes))


for param in model.vit.parameters():
    param.requires_grad = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

with open("training_metrics.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])


    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=10,
        device=device
    )

    for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
            results["train_loss"], results["train_acc"],
            results["val_loss"], results["val_acc"])):
        writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])


torch.save(model.state_dict(), "fine_tuned_model.pth")
print("Model saved successfully as fine_tuned_model.pth!")


print("Evaluating on the test set...")
model.eval()
y_true = []
y_pred = []


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)["logits"]
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_loader.dataset.classes)
disp.plot(cmap=plt.cm.Blues)


plt.title("Confusion Matrix for Test Evaluation")
plt.savefig("confusion_matrix.png")
plt.show()

print("Test evaluation completed. Confusion matrix saved as 'confusion_matrix.png'.")

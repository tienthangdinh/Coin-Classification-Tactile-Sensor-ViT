import torch
from torch import nn
from dataloader import get_dataloaders
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score


_, _, test_loader = get_dataloaders(batch_size=32)

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)

model.classifier = nn.Linear(model.config.hidden_size, len(test_loader.dataset.classes))

model.load_state_dict(torch.load("fine_tuned_model.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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


accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

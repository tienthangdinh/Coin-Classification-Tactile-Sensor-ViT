from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),        
])

train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset/val", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)


def get_dataloaders(batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

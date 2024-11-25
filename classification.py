from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# preprocessing
data_transforms = {
    'images': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'images': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# load data
data_dir = 'ISS-Dataset/2D_and_3D_in_carla/2Ddataset'
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
                  for x in ['images', 'images']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['images', 'images']}

# load model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(image_datasets['images'].classes))  # 修改输出层

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10):  # 10 epochs
    print(f'Epoch {epoch + 1}/10')
    for phase in ['images', 'images']:
        model.train() if phase == 'images' else model.eval()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'images'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'images':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(torch.argmax(outputs, 1) == labels)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = corrects.double() / len(image_datasets[phase])
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

torch.save(model.state_dict(), 'datasets/classification_model.pth')  # save model

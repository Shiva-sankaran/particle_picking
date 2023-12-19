from src.DRPnet import DRPDetNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

lr = 1e-3
batch_size = 32
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_data = np.load("/home/shiva/projects/particle_picking/DRPnet-py/data/10005/np_training_data/orig_patch_data.npy")
target_labels = np.load("/home/shiva/projects/particle_picking/DRPnet-py/data/10005/np_training_data/labels_patch_data.npy")
input_data = np.transpose(input_data, (3, 2, 0, 1))
target_labels = np.transpose(target_labels, (3, 2, 0, 1))
input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
target_labels = torch.tensor(target_labels, dtype=torch.long).to(device)

print("Input data shape: ",input_data.shape)

print("Target data shape: ",target_labels.shape)



train_data, val_data, train_labels, val_labels = train_test_split(input_data, target_labels, test_size=0.2, random_state=42)

print(train_data.shape)
print(val_data.shape)

print(train_labels.shape)
print(val_labels.shape)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


model = DRPDetNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)


# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.to(torch.float32)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() #* inputs.size(0)
    epoch_loss = running_loss / len(train_loader) # inputs.shape[2] * inputs.shape[3] 
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}")

    # Validation loop after each epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() #* inputs.size(0)
        val_loss /= len(val_loader) #inputs.shape[2] * inputs.shape[3] 
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation MSE Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

save_path = '/home/shiva/projects/particle_picking/DRPnet-py/ckpts/model_checkpoint.pth'
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': epoch_loss,
}, save_path)

print("Done training")







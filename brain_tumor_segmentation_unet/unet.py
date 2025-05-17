import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
import torch
import torch.nn as nn
import torch.optim as optim

import dataset

# 1. Define model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,   # T1
    classes=1,       # binary mask (tumor or not)
)

# 2. Define preprocessing, loss, metrics
loss = DiceLoss(mode='binary')
metrics = [IoU(threshold=0.5)]

# 3. Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. Prepare data loaders
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 5. Define training epochs
train_epoch = TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
)

valid_epoch = ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
)

# 6. Run training
for i in range(10):  # 10 epochs
    print(f'\nEpoch {i+1}')
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CaptionDataset
from lr_scheduler import CosineScheduler
from model import CaptionModel


batch_size = 64
epochs = 20
min_lr = 3e-6
lr = 6e-4
workers = 2
grad_clip = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop((128, 128), scale=(0.6, 1)),
        transforms.RandomRotation(15),
    ]
)
ds = CaptionDataset("result.csv", "flickr30k_images", transform=transform)
train_loader = DataLoader(ds, batch_size, shuffle=True, workers=workers)
epoch_iters = len(train_loader)
warmup_iters = 3 * epoch_iters
decay_iters = (epochs - 1) * epoch_iters

model = CaptionModel()
model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
scheduler = CosineScheduler(
    optimizer,
    warmup_iters,
    decay_iters,
    min_lr,
    lr,
)

for epoch in tqdm(range(epochs)):
    train_loss = 0.0
    for iter, (image, text) in enumerate(train_loader):
        image = image.to(device)
        text = text.to(device)
        scheduler.step()
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            pred = model(image, text)
            loss = criterion(pred, text)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        if iter % (epoch_iters // 5) == 0:
            print(
                f"Epochs: {epoch}/{epochs} | Iters: {iter}/{epoch_iters} | Train_loss {train_loss / iter:.4f}"
            )
torch.save(model.state_dict(), "model.pt")

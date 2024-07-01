import numpy as np
import pandas as pd
import torch
import csv
import torch.optim as optim
import sys

from dataset import dataset
from vae import VAE

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# 学習データ
x_train = np.load('./data/x_train.npy')
# テストデータ
x_test = np.load('./data/x_test.npy')

trainval_data = dataset(x_train)
test_data = dataset(x_test)

batch_size = 32

val_size = 10000
train_size = len(trainval_data) - val_size

train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = 22
n_epochs = 100
patience = 10
model = VAE(z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=round(0.00295482955411945,6))
best_loss = np.inf
patience_counter = 0

for epoch in range(n_epochs):
    losses = []
    KL_losses = []
    reconstruction_losses = []
    model.train()
    for x in dataloader_train:

        x = x.to(device)

        model.zero_grad()

        # KL_loss, reconstruction_lossの各項の計算
        KL_loss, reconstruction_loss = model.loss(x)

        # エビデンス下界の最大化のためマイナス付きの各項の値を最小化するようにパラメータを更新
        loss = KL_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        KL_losses.append(KL_loss.cpu().detach().numpy())
        reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

    losses_val = []
    model.eval()
    for x in dataloader_valid:

        x = x.to(device)
        KL_loss, reconstruction_loss = model.loss(x)
        loss = KL_loss + reconstruction_loss

        losses_val.append(loss.cpu().detach().numpy())

    valid_loss = np.mean(losses_val)

    # 早期終了のロジック
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), './best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print('EPOCH:%d, Train Lower Bound:%lf, (%lf, %lf), Valid Lower Bound:%lf' %
          (epoch+1, np.average(losses), np.average(KL_losses), np.average(reconstruction_losses), np.average(losses_val)))
    sys.stdout.flush()
    
sample_x = []
answer = []
model.load_state_dict(torch.load('./best_model.pth'))
model.eval()
for x in dataloader_test:

    x = x.to(device)

    y, _ = model(x)

    y = y.view(y.size(0), -1)

    y = y.tolist()

    sample_x.extend(y)

with open('./submission_pred.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(sample_x)
file.close()
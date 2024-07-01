import optuna
import numpy as np
import pandas as pd
import torch
import csv
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import dataset
from vae import VAE
import sys
from setproctitle import setproctitle

# 学習データ
x_train = np.load('./data/x_train.npy')
# テストデータ
x_test = np.load('./data/x_test.npy')

trainval_data = dataset(x_train)
test_data = dataset(x_test)

batch_size = 32
val_size = 10000
train_size = len(trainval_data) - val_size

train_data, val_data = random_split(trainval_data, [train_size, val_size])

dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(val_data, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 10

def objective(trial):
    # Optunaでパラメータをサンプリング
    z_dim = trial.suggest_int('z_dim', 5, 50)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    lr = round(lr, 6)

    model = VAE(z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        losses = []
        model.train()
        for x in dataloader_train:
            x = x.to(device)
            model.zero_grad()
            KL_loss, reconstruction_loss = model.loss(x)
            loss = KL_loss + reconstruction_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        # 検証データに対する評価
        losses_val = []
        model.eval()
        with torch.no_grad():
            for x in dataloader_valid:
                x = x.to(device)
                KL_loss, reconstruction_loss = model.loss(x)
                loss = KL_loss + reconstruction_loss
                losses_val.append(loss.cpu().detach().numpy())

        val_loss = np.mean(losses_val)
        print(f'EPOCH:{epoch+1}, Train Loss:{np.mean(losses)}, Valid Loss:{val_loss}')
        sys.stdout.flush()

    return val_loss

if __name__ == "__main__":
    setproctitle("VAE工事中")
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')

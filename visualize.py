import numpy as np
import torch
import torch.optim as optim
import sys
from PIL import Image
import os

from dataset import dataset
from vae import VAE

def save_image(tensor, save_path):
    img = tensor.detach().cpu().numpy() * 255
    img = img.astype(np.uint8)
    
    # Assuming tensor shape is (batch_size, channels, height, width)
    # Take the first image in the batch
    img = img[0]
    
    # If the image has a single channel, convert it to a 2D array
    if img.shape[0] == 1:
        img = img[0]
    
    img = Image.fromarray(img)
    img.save(save_path)

def main(opt):

    model_path = opt.model_path

    # テストデータ
    x_test = np.load('./data/x_test.npy')

    test_data = dataset(x_test)

    dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z_dim = 22
    model = VAE(z_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    for i, x in enumerate(dataloader):
        x = x.to(device)
        x_recon, _ = model(x)

        # Create directory based on original image index
        dir_path = os.path.join('./result', f'image_{i}')
        os.makedirs(dir_path, exist_ok=True)

        # Save the input and reconstructed images
        save_image(x, os.path.join(dir_path, 'input.png'))
        save_image(x_recon, os.path.join(dir_path, 'output.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    opt = parser.parse_args()
    main(opt)

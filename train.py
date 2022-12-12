import argparse
import time 
from tqdm import tqdm 
import torch
from torch import optim
import numpy as np

from dataset import FloodDataLoader, preprocessed_split_data
from custom_metrics import compute_MSE_per_step, compute_r2_metrics
from model import FloodModel


def do_compute(model, batch, device):
    batch = batch.to(device=device)
    out = model(batch)

    return  out


def run_batch(model, optimizer, data_loader, epoch_i, desc, device):
        total_loss = 0
        MSE_list = []
        R2_list = []
        
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            wdfp_out, (v_norm_out, v_out), loss  = do_compute(model, batch, device)

            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                targets = batch.wdfp.cpu().squeeze(-1).numpy()[:, 1:]
                preds = wdfp_out.cpu().squeeze(-1).numpy()
                MSE_list.append(compute_MSE_per_step(targets, preds))
                R2_list.append(compute_r2_metrics(targets, preds))

        total_loss /= len(data_loader)
        MSE_list = np.stack(MSE_list, axis=0)
        rmse_metrics = np.sqrt(np.mean(MSE_list, axis=0))
        r2_score = np.stack(R2_list, axis=0).mean(0)

        return total_loss, rmse_metrics, r2_score


def print_metrics(loss, rmse_metrics, r2_score):
    print(f'loss: {loss:.4f}, rmse: {rmse_metrics:.4f}, r2: {r2_score:.4f}')


def train(model, train_data_loader, val_data_loader, optimizer, n_epochs, device):

    for epoch_i in range(1, n_epochs+1):
        
        start = time.time()
        model.train()
        ## Training
        train_loss, train_rmse, train_r2 = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', device)

        model.eval()
        with torch.no_grad():
            ## Validation 
            if val_data_loader:
                val_loss , val_rmse,  val_r2= run_batch(model, optimizer, val_data_loader, epoch_i, 'val', device)

        if train_data_loader:
            print(f'\n#### Epoch {epoch_i} time {time.time() - start:.4f}s')
            print_metrics(train_loss, train_rmse.mean(), train_r2.mean())

        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, val_rmse.mean(), val_r2.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--time_steps', type=int, default=10, help='Total number of time steps.')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--v_h_dim', type=int, default=32, help='Vector features - hidden dimensions.')
    parser.add_argument('--s_h_dim', type=int, default=32, help='Scalar features - hidden dimensions.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_conv_layers', type=int, default=3)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, valid_dataset, test_dataset, test_indices = preprocessed_split_data(args.time_steps, args.test_ratio, args.val_ratio)

    train_loader = FloodDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    valid_loader = FloodDataLoader(valid_dataset, batch_size=args.batch_size) 

    sample = train_dataset[0]
    s_in_dim = sample['scalar'].shape[-1] + sample['wdfp'].shape[-1] + sample['scalar_fixed'].shape[-1]
    v_in_dim = sample['vector'].shape[-2]
    args.in_dims = (int(s_in_dim), int(v_in_dim))
    ###================
    model = FloodModel(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(model.__class__.__name__)
    model.to(device=device)
    print(args)
    print(f'Train on {len(train_dataset)}, Validating on {len(valid_dataset)}')
    print('Training on', device)

    train(model, train_loader, valid_loader, optimizer, args.epochs, device)


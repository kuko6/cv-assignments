import os
import glob
import numpy as np
import wandb
import copy
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary

from utils import EarlyStopper, SignDataset
from model import Model


config = {
    "lr": 1e-2, # 1e-3
    "dataset": "ASL",
    "epochs": 20,
    "batch_size": 256,
    "image_shape": (64, 64, 3),
    "classes": 29,
    "activations": "ReLU",
    "conv_layers": 4,
    "loss": "cross-entropy",
    "optimizer": "Adam",
    "augment": True,
    "scheduler": True,
    "earlyStoppingValue": "loss" # loss/accuracy
}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


def prepare_data(data_dir):
    data = glob.glob('*/*.[jp][pn]g', root_dir=data_dir)
    print(len(data))
    print(data[:5])

    labels = sorted(glob.glob('*', root_dir=data_dir))
    print(len(labels))

    labels_map = dict()
    reversed_labels_map = dict()
    for i, label in enumerate(labels):
        labels_map[i] = label
        reversed_labels_map[label] = i

    data_train, data_val = train_test_split(data, test_size=0.2, train_size=0.8, random_state=420)
    print(f'train: {len(data_train)}, val: {len(data_val)} | {len(data_train)+len(data_val)}')

    transform = transforms.Compose([transforms.Resize(config['image_shape'][0]+4, antialias=False), transforms.CenterCrop(config['image_shape'][0])])
    train_data = SignDataset(data_train, data_dir, reversed_labels_map, transform=transform, augment=config['augment'])
    val_data = SignDataset(data_val, data_dir, reversed_labels_map, transform=transform, augment=False)

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, labels_map


def calculate_acc(y_pred, y):
    preds = torch.argmax(y_pred, dim=1)
    num_correct = (preds == y).sum().item()

    return num_correct / len(y)


def preview(model: Model, dataloader: DataLoader, filename, labels_map, save=False, use_wandb=False, step=None):
    model.eval()
    with torch.no_grad():
        fig, axs = plt.subplots(3, 6, figsize=(20, 13))
        axs = axs.flatten()
        
        images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        
        y_pred = model(images)
        y_pred_labels = torch.argmax(y_pred, dim=1) 
        for i, (img, label) in enumerate(zip(images, labels)):
            if i == 18: break

            original = f'Original: {label.item()}, {labels_map[label.item()]}'
            predicted = f'Predicted: {y_pred_labels[i].item()}, {labels_map[y_pred_labels[i].item()]}'
            
            title_color = 'black' if labels[i].item() == y_pred_labels[i].item() else 'red' 
            axs[i].imshow(img[:,:,:].cpu().permute(1, 2, 0))
            axs[i].set_title(original+'\n'+predicted, fontsize=9, color=title_color)
            axs[i].axis('off')
            # plt.show()
         
        if save:
            fig.savefig(f'outputs/{filename}_preview.png')
            plt.close(fig)
        
    if use_wandb:
        wandb.log({'preview': wandb.Image(f'outputs/{filename}_preview.png')}, step=step)  


def train_one_epoch(device, dataloader, model, loss_fn, optimizer):
    model.train()
    avg_loss, avg_acc = 0, 0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        # Get prediction 
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)
        avg_loss += loss.item()
        
        # Compute acc
        acc = calculate_acc(y_pred, y)
        avg_acc += acc
        
        # Update parameters
        loss.backward()
        optimizer.step()

        print(f'training step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, acc: {acc:>5f}', end='\r')
    
    print()

    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader)
        
    return (avg_loss, avg_acc)


def validate(device, dataloader, model, loss_fn, epoch, labels_map, use_wandb):    
    model.eval()
    avg_loss, avg_acc = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            loss = loss_fn(y_pred, y).item()
            avg_loss += loss
            
            acc = calculate_acc(y_pred, y)
            avg_acc += acc

            print(f'validation step: {i+1}/{len(dataloader)}, loss: {loss:>5f}, acc: {acc:>5f}', end='\r')

    print()
    
    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader)

    preview(model, dataloader, epoch, labels_map=labels_map, save=True, use_wandb=use_wandb, step=epoch)
    
    return (avg_loss, avg_acc)


def train(train_dataloader, val_dataloader, labels_map, model, optimizer, use_wandb):
    loss_fn = nn.CrossEntropyLoss()

    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)

    early_stopper = EarlyStopper(patience=5, delta=0, mode='min')
    
    epochs = config['epochs']
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}
    best_checkpoint = {'epoch': 0, 'val_acc': 0, 'val_loss': np.inf, 'model_state': None, 'optimizer_state': None}

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        
        train_loss, train_acc = train_one_epoch(device, train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = validate(device, val_dataloader, model, loss_fn, epoch, labels_map, use_wandb)

        print('-------------------------------')
        
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)

        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)

        if config['scheduler']:
            scheduler.step(val_loss)
        
        print(f'loss: {train_loss:>5f} acc: {train_acc:>5f}')
        print(f'val loss: {val_loss:>5f} val acc: {val_acc:>5f}')
        # print(f'lr: {optimizer.param_groups[0]["lr"]:>5f}')
        print('===============================')

        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()
        }, 'outputs/checkpoint.pt')

        if best_checkpoint['val_acc'] < val_acc:
            best_checkpoint['epoch'] = epoch
            best_checkpoint['val_acc'] = val_acc
            best_checkpoint['val_loss'] = val_loss
            best_checkpoint['model_state'] = model.state_dict()
            best_checkpoint['optimizer_state'] = optimizer.state_dict()
            torch.save(best_checkpoint, 'outputs/best_checkpoint.pt')

        if use_wandb:
            wandb.log({
                'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc, 
                'val_loss':val_loss, 'val_accuracy': val_acc, 'lr': optimizer.param_groups[0]["lr"]
            }, step=epoch)

        if early_stopper(val_loss):
            print('Stopping early!!!')
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to content dataset')
    parser.add_argument('--wandb', type=str, help='wandb id')
    parser.add_argument('--model_path', type=str, help='path to model')
    args = parser.parse_args()

    use_wandb = False
    wandb_key = args.wandb
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="assignment-3", name="", reinit=True, config=config)    
        use_wandb = True

    if args.content_path and args.style_path and args.preview_path:
        data_dir = args.content_path
        style_dir = args.style_path
        preview_dir = args.preview_path
    else:
        print('You didnt specify the data path >:(')
        return
    
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    model = Model()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['lr'])
    if args.model_path:
        # From checkpoint
        checkpoint = torch.load('outputs/checkpoint.pt')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        config['max_iter'] -= checkpoint['iter']

        # From final model
        # model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    # print(summary(model))
    model.to(device)

    train_dataloader, val_dataloader, labels_map = prepare_data(data_dir)

    train(train_dataloader, val_dataloader, labels_map, model, optimizer, use_wandb)

    torch.save(model.state_dict(), 'outputs/model.pt')
    if use_wandb:
        model_artifact = wandb.Artifact('final_model', type='model')
        model_artifact.add_file('outputs/model.pt')
        wandb.log_artifact(model_artifact)

        best_checkpoint_artifact = wandb.Artifact('best_checkpoint', type='checkpoint')
        best_checkpoint_artifact.add_file('outputs/best_checkpoint.pt')
        wandb.log_artifact(best_checkpoint_artifact)
        wandb.finish()


if __name__ == '__main__':
    main()
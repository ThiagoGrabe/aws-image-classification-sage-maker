#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os
import json
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
                                                                                )
    )
    return test_loss

def train(model, train_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data=data.to(device)
        target=target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    
def net(n_features):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    model = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, n_features)

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.2859, std=0.3530),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.2859, std=0.3530),
    ])

    
    train_dataset = torchvision.datasets.FashionMNIST(os.path.join(data,'.pytorch/train'), transform=train_transform)
    test_dataset = torchvision.datasets.FashionMNIST(os.path.join(data,'.pytorch/test'), transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
    
    return train_loader, test_loader


def save_model(model, model_path):
    path = os.path.join(model_path, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    
    model=net(args.n_features)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook=''
    
    train_loader, test_loader = create_data_loaders(args.data_dir, 
                                                    args.batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    print("\nHyperparameters [Learning Rate {:.6e}, eps {:.6e}, weight decay {:.4e}\n".format(args.lr, args.eps, args.weight_decay))
    
    for epoch in range(1, args.epochs + 1):
        print("\nEpoch: {}\nTraining".format(epoch))
        train(model, train_loader, loss_criterion, optimizer, device, hook)
        print("\nEpoch: {}\nTesting".format(epoch))
        loss = test(model, test_loader, loss_criterion, device, hook)
    
    '''
    Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, 'model.pth'))
    torch.save(model.state_dict(), "model.pt")
    save_model(model, model_path=args.model_dir)

if __name__=='__main__':
    
    
    '''
    TODO: Specify any training args that you might need
    '''
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="EP",
        help="number of epochs to train (default: 10)",
    )
    
    parser.add_argument(
        "--n_features",
        type=int,
        default=10,
        metavar="EP",
        help="number of output features (default: 10)",
    )
    
    parser.add_argument(
        '--no-cuda', 
        action='store_true', 
        default=False,
        help='disables CUDA training')

    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        metavar="LR", 
        help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--eps", 
        type=float, 
        default=1e-8, 
        metavar="EPS", 
        help="eps (default: 1e-8)"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=1e-2, 
        metavar="WEIGHT-DECAY", 
        help="weight decay coefficient (default 1e-2)"
    )
    
    parser.add_argument(
        "--hosts", 
        type=list, 
        default=json.loads(os.environ["SM_HOSTS"])
    )
    
    parser.add_argument(
        "--current-host", 
        type=str, 
        default=os.environ["SM_CURRENT_HOST"]
    )
    
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"]
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=os.environ["SM_CHANNEL_TRAINING"]
    )
    
    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=os.environ["SM_NUM_GPUS"]
    )
    
    parser.add_argument(
        "--num-cpus", 
        type=int, 
        default=os.environ["SM_NUM_CPUS"]
    )
    
    args=parser.parse_args()
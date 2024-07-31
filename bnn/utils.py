import os
import torch
import tabulate
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import partial
from time import perf_counter
import numpy as np
from data.load_hf_datasets import get_ds, collate_fn, data_preprocess
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

device = "cuda" if torch.cuda.is_available() else "cpu"
import random

torch.backends.cudnn.deterministic = True
def set_seed(seed, cuda):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def adjust_learning_rate(args, optimizer, epoch, batch_idx, num_batch, T):
    if args.lr_type == 'cyclic':
        rcounter = epoch*num_batch+batch_idx
        cos_inner = np.pi * (rcounter % (T // args.M))
        cos_inner /= T // args.M
        cos_out = np.cos(cos_inner) + 1
        factor = 0.5*cos_out
    else:
        t = (epoch) / args.epochs 
        lr_ratio = 0.01       
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
    lr = args.lr_init * factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def run_epoch(args,loader, model, criterion, epoch, num_batch=None, T=None, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase == "train":
        model.train()
    elif phase == "eval":
        model.eval()

    ttl = 0
    start = perf_counter()
    with torch.autograd.set_grad_enabled(phase == "train"):

        for i, (input, target) in enumerate(loader):
            target = target.to(device=device)
            if phase == "train":
                lr = adjust_learning_rate(args, optimizer, epoch,i,num_batch, T)
            input = input.to(device=device)
            output = model(input)
            pred = output.data.max(1, keepdim=True)[1] 
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * target.size(0)       
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += target.size(0)
            if phase == "train":            
                optimizer.zero_grad()
                loss.backward()
                if args.set_default_optimizer:                    
                    optimizer.step(epoch)
                else:
                    optimizer.step(epoch)
            
            if i % 100 == 0:
                print(f'{i // 100}', end=' - ')

    elapse = perf_counter() - start
    correct = correct.cpu().item()
    res = {
        "loss": loss_sum / float(ttl),
        "accuracy": correct / float(ttl) * 100.0,
    }
    if phase == "train":
        res["train time"] = elapse

    return res


def print_table(columns, values, epoch):
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


num_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
}


def get_data(dataset, data_path, batch_size, num_workers,train_shuffle=True):
    print("Loading dataset {} from {}".format(dataset, data_path))
    if dataset in ["CIFAR10", "CIFAR100"]:
        ds = getattr(datasets, dataset.upper())
        path = os.path.join(data_path)
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_set = ds(path, train=True, download=True, transform=transform_train)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        loaders = {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=train_shuffle,
                num_workers=0
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            ),
        }
    elif dataset == 'IMAGENET1K':
        path = os.path.join(data_path)
        dataset_train, dataset_val = get_ds("imagenet-1k", data_path, num_proc=4)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32),  
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),             
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  
        ])

        
        test_transforms = transforms.Compose([
            transforms.Resize(34),    
            transforms.CenterCrop(32), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  
        ])
        dataset_train.set_transform(partial(data_preprocess, transform_f=train_transforms))
        dataset_val.set_transform(partial(data_preprocess, transform_f=test_transforms))
        loaders = {
            "train": torch.utils.data.DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=train_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            ),
            "test": torch.utils.data.DataLoader(
                dataset_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            ),
        }

    else:
        raise Exception("Invalid dataset %s" % dataset)    

    return loaders
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from jittor.models import Resnet18, Resnet34
from tqdm import tqdm
from jittor.models.resnet import *
import os
import numpy as np
from PIL import Image
import argparse
import random

jt.flags.use_cuda = 1


# ============== Dataset ==============
class ImageFolder(Dataset):
    def __init__(self, root, annotation_path=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
            data_dir = [(x[0], int(x[1])) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        return jt.array(image), label

# ============== Model ==============
class Net(nn.Module):
    def __init__(self, num_classes, pretrain):
        super().__init__()

        self.base_net = Resnet34(num_classes=num_classes, pretrained=pretrain) # SGD 0.8335

    def execute(self, x):
        x = self.base_net(x)
        return x

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, now_epoch:int, num_epochs:int):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))
    step = 0
    for data in pbar:
        step += 1
        image, label = data
        pred = model(image)
        loss = nn.cross_entropy_loss(pred, label, weight=class_weights)
        loss.sync()
        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.2f}')

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model:nn.Module, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    print("Evaluating...")
    for data in val_loader:
        image, label = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        targets.append(label.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(np.float32(preds == targets))
    return acc

def run(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, num_epochs:int, modelroot:str):
    best_acc = 0
    for epoch in range(num_epochs):
        training(model, optimizer, train_loader, epoch, num_epochs)
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(modelroot, 'best.pkl'))
        print(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')

# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    for data in test_loader:
        image, image_names = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')

# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./TrainSet')
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='./model_save/best.pkl')
    parser.add_argument('--result_path', type=str, default='./result.txt')
    args = parser.parse_args()

    model = Net(pretrain=True, num_classes=6)
    transform_train = Compose([
        Resize((512, 512)),
        RandomCrop(448),
        RandomHorizontalFlip(),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((512, 512)),
        CenterCrop(448),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

   # ─────── 1. 定义每个类别的样本数 ───────
    samples_per_cls = [364, 608, 216, 16, 6, 291]
    beta = 0.005
    eff_num = 1.0 - np.power(beta, np.array(samples_per_cls))
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.sum() * len(samples_per_cls)
    class_weights = jt.array(weights, dtype=jt.float32)


    if not args.testonly:

        optimizer = nn.SGD(model.parameters(), lr=0.0021, momentum=0.9)

        train_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/train.txt'),
            transform=transform_train,
            batch_size=8,
            num_workers=8,
            shuffle=True
        )
        val_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
            transform=transform_val,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )
        run(model, optimizer, train_loader, val_loader, 1000, args.modelroot)
    else:
        test_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'TestSetA'),
            transform=transform_val,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )
        model.load(args.loadfrom)
        test(model, test_loader, args.result_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import time
from model import ResNet18, Vgg16_Net
from utils import *
 
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='AI Sec homework_1')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar-10')
parser.add_argument('--epochs', type=int, default=40, help="epochs")
parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="L2")
parser.add_argument('--scheduler', type=int, default=1, help="lr scheduler")
parser.add_argument('--seed', type=int, default=0, help="lucky number")
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
print("Running on %s" % device)

#net = ResNet18().to(device)
net = Vgg16_Net().to(device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
 
trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

x = []
loss_list = []
train_acc_list = []
test_acc_list = []
iter_count = 0

start_time = time.time()
for epoch in range(args.epochs):
    print('epoch: %d' % (epoch + 1))

    net.train()
    for i, data in enumerate(trainloader, 0):
                     
        inputs, labels = data
        optimizer.zero_grad()
        pred = net(inputs.to(device))
        loss = criterion(pred, labels.to(device))
        loss.backward()
        optimizer.step()


        if iter_count % 100 == 0:

            train_acc = test_accuracy(trainloader, net, 1000)
            test_acc = test_accuracy(testloader, net, 1000)
            print("iter:",iter_count,"loss: %.3f train acc: %.3f test acc: %.3f" % (loss.item(), train_acc, test_acc))
        
            x.append(iter_count)
            loss_list.append(loss.item())
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
        
        iter_count += 1

    if args.scheduler:
        scheduler.step()

    plt.plot(x, train_acc_list, label = "train_acc")
    plt.plot(x, test_acc_list, label = "test_acc")
    plt.grid()
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.savefig("img/train_acc.png")
    plt.close("all")

    plt.plot(x, loss_list)
    plt.grid()
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig("img/train_loss.png")
    plt.close("all")
train_time = time.time() - start_time
print("train time:", train_time)


import os 
import time
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets

import fused_resnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--data_root", type=str, default="../../../data")
    parser.add_argument("--dataset", type=str, default="MNIST", help="MNIST, CIFAR-10, N-MNIST, DvsGesture")
    parser.add_argument("--arch", type=str, default="Spiking-ResNet18", help="Spiking-ResNet18, Spiking-ResNet34, Spiking-ResNet50")
    parser.add_argument("--timing", action="store_true", default=True, help="Timing or not")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda:0"

    T  = 32    # time steps
    E  = 5     # number of epochs
    lr = 1e-3  # learning rate
    
    
    """ 
        Load the training and testing datasets 
    """
    if args.dataset == "MNIST":
        C = 10    # number of classes
        B = 100   # batch size
        trainset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True)
        testset = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)
        is_dvs = False
    
    elif args.dataset == "CIFAR-10":
        C = 10    # number of classes
        B = 100   # batch size
        trainset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True)
        testset = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)
        is_dvs = False
    
    elif args.dataset == "N-MNIST":
        from spikingjelly.datasets.n_mnist import NMNIST
        C = 10    # number of classes
        B = 100   # batch size
        trainset = NMNIST(root=os.path.join(args.data_root, "NMNIST"), train=True, data_type="frame", frames_number=T, split_by="number")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True)
        testset = NMNIST(root=os.path.join(args.data_root, "NMNIST"), train=False, data_type="frame", frames_number=T, split_by="number")
        testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)
        is_dvs = True
    
    elif args.dataset == "DvsGesture":
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        C = 11    # number of classes
        B = 10    # batch size
        trainset = DVS128Gesture(root=os.path.join(args.data_root, "DvsGesture"), train=True, data_type="frame", frames_number=T, split_by="number")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True)
        testset = DVS128Gesture(root=os.path.join(args.data_root, "DvsGesture"), train=False, data_type="frame", frames_number=T, split_by="number")
        testloader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False)
        is_dvs = True
    
    else:
        print("Error: Not Supported Dataset.")
        import sys; sys.exit(1)
    
    
    """
        Model Selection
    """
    if args.arch == "Spiking-ResNet18":
        model = fused_resnet.spiking_resnet18(time_step=T, num_classes=C).to(device)
    
    elif args.arch == "Spiking-ResNet34":
        model = fused_resnet.spiking_resnet34(time_step=T, num_classes=C).to(device)
    
    elif args.arch == "Spiking-ResNet50":
        model = fused_resnet.spiking_resnet50(time_step=T, num_classes=C).to(device)
    
    else:
        print("Error: Not Supported Model Selection.")
        import sys; sys.exit(1)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(model)
    print(f"selected dataset: {args.dataset}")
    print(f"selected model:   {args.arch}")
    
    
    """
        Training
    """
    for epoch in range(E):
        running_loss = 0.0
        model.train()
        t0 = time.time()
        for index, (images, labels) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            if is_dvs:
                images = images.permute(1, 0, 2, 3, 4)
                images = torch.cat([images, torch.zeros_like(images[:, :, :1])], dim=2).to(device)
            else:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                images = torch.stack([(images > torch.rand(images.shape)).float() for _ in range(T)]).to(device)
            logits = model(images)
            logits = logits.mean(dim=0) 
            batch = min(images.shape[1], B)
            labels = torch.zeros(batch, C).scatter_(1, labels.view(-1, 1), 1).to(device)
            loss = criterion(logits, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
    
            if not args.timing and index % 100 == 0:
                print("[%d, %5d] loss: %.5f" %(epoch + 1, index, running_loss))
                running_loss = 0.0
        torch.cuda.synchronize()
        print(f"Epoch {epoch}, Training time elapsed: {time.time() - t0}")
    
    
        """
            Validating
        """
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            t1 = time.time()
            for (images, labels) in testloader:
                if is_dvs:
                    images = images.permute(1, 0, 2, 3, 4)
                    images = torch.cat([images, torch.zeros_like(images[:, :, :1])], dim=2).to(device)
                else:
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1)
                    images = torch.stack([(images > torch.rand(images.shape)).float() for _ in range(T)]).to(device)
                logits = model(images)
                logits = logits.mean(dim=0) 
                _, predicted = torch.max(logits.data, 1)
                labels = labels.to(device)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            torch.cuda.synchronize()
            print(f"Epoch {epoch}, Testing time elapsed:  {time.time() - t1}")
        
        print("Epoch %d, Accuracy of the network on the test images: %.6f %%" % (epoch, 100 * correct / total))
    
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchaudio.datasets as datasets
from dataset import audioDataset
import pandas as pd
import os
import torchvision
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np



SAMPLE_RATE = 16000 
NUM_SAMPLES = 32000 
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

DATASET = "emotion" # emotion or gender
CNN = "resnet50" # densenet121, alexnet, mobilenetv3 or resnet50

if DATASET == "gender":
    VALIDATION_AUDIO_DIR = "H:\\voxc2\\voxceleb2\\test\\aac"
    AUDIO_DIR = "H:\\voxc2\\voxceleb2\\dev\\aac"
if DATASET == "emotion":
    AUDIO_DIR = "EMOTION_TRAIN"
    VALIDATION_AUDIO_DIR = "EMOTION_TEST"

def create_data_loader(train_data, batch_size, datasampler):
    if datasampler == None:
        train_dataloader = DataLoader(train_data, batch_size)
    else: 
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=datasampler)

    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    class_labels = []
    loss_avg = []
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        loss = loss_fn(prediction, target)
        loss_avg.append(loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Training loss: {sum(loss_avg) / len(loss_avg)}")
    return (sum(loss_avg) / len(loss_avg)).item()

def get_validation_loss(model, data_loader, loss_fn, device):
    loss_avg = []
    with torch.no_grad():
        model.eval()
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            loss = loss_fn(prediction, target)
            loss_avg.append(loss)
        model.train()
        print(f"Validation loss: {sum(loss_avg) / len(loss_avg)}")
    return (sum(loss_avg) / len(loss_avg)).item() #.data.numpy()

# validation accuracy during training
def check_accuracy(model, testloader, device, loss_fn):
    total = 0
    correct = 0
    loss_avg = []
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            labels = torch.tensor(labels)
            labels = labels.cuda()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.unsqueeze_(0))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    model.train()
    print('Validation Accuracy of the network: %d %%' % (100 * correct / len(testloader)))
    return 100 * correct / len(testloader)

# training accuracy during training
def check_train_accuracy(model, testloader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            labels = torch.tensor(labels)
            labels = labels.cuda()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.unsqueeze_(0))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    model.train()
    print('Training Accuracy of the network: %d %%' % (100 * correct / len(testloader)))
    return 100 * correct / len(testloader)

def train(model, data_loader, loss_fn, optimiser, device, scheduler, epochs, validation_dataloader):
    since = time.time()
    loss_values = []
    val_loss_values = []
    epoch_amount = []
    val_accuracy = []
    train_accuracy = []
    lowest_val = float('inf')
    lowest_val_acc = float("-inf")
    for i in range(epochs):
        print(f"Epoch {i+1}")
        loss = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        val_loss = get_validation_loss(model, data_loader, loss_fn, device)
        t_acc = round(check_train_accuracy(model, trainingSubset, device),5)
        v_acc = round(check_accuracy(model, validationData, device, loss_fn),5)
        train_accuracy.append(t_acc)
        val_accuracy.append(v_acc)
        loss_values.append(round(loss, 5))
        val_loss_values.append(round(val_loss, 5))
        epoch_amount.append(i+1)
        if (val_loss < lowest_val):
            print("new lowest loss")
            lowest_val = val_loss
            torch.save(cnn.state_dict(), f"models/{CNN}_{DATASET}_epoch_{i+1}.pth")
            print("model saved")
        if (v_acc > lowest_val_acc):
            print("new highest accuracy")
            lowest_val_acc = v_acc
            torch.save(cnn.state_dict(), f"models/{CNN}_{DATASET}_epoch_{i+1}.pth")
            print("model saved")

        print('Epoch completed in {:.0f}m {:.0f}s'.format((time.time() - since) // 60, (time.time() - since) % 60))

        print("---------------------------")
        scheduler.step()
        
    torch.save(cnn.state_dict(), f"models/{CNN}_{DATASET}_final.pth")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Finished training")

# plots
    with torch.no_grad():
        plt.plot(epoch_amount,loss_values, label="Training loss")
        plt.plot(epoch_amount,val_loss_values, label="Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(np.arange(min(epoch_amount), max(epoch_amount)+1, 1))
        plt.legend()
        plt.show()
        plt.plot(epoch_amount, val_accuracy, label="Validation accuracy")
        plt.plot(epoch_amount, train_accuracy, label="Training accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(np.arange(min(epoch_amount), max(epoch_amount)+1, 1))
        plt.ylim(10,100)
        plt.legend()
        plt.show()
    print("loss values: ", loss_values)
    print("validation loss values: ", val_loss_values)
    print("epoch amount: ", epochs)
    print("training accuracy: ", train_accuracy)
    print("validation accuracy:", val_accuracy)

cuda_ = "cuda:0"
if __name__ == "__main__":
    
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        
    )
    # creating training and validation custom datasets
    emotionData = audioDataset(
                            AUDIO_DIR,
                            None, 
                            mel_spectrogram,
                            NUM_SAMPLES, 
                            SAMPLE_RATE,
                            device,
                            DATASET)
    validationData = audioDataset(
                            VALIDATION_AUDIO_DIR,
                            None, 
                            mel_spectrogram,
                            NUM_SAMPLES, 
                            SAMPLE_RATE,
                            device,
                            DATASET)
    trainingSubset = emotionData

    train_dataloader = create_data_loader(emotionData, BATCH_SIZE, None) 
    validation_dataloader = create_data_loader(validationData, BATCH_SIZE, None) 

    # choosing correct cnn
    if CNN == "densenet121":
        cnn = torchvision.models.densenet121(pretrained=True)
        print(cnn.classifier)
        if DATASET == "gender":
            cnn.classifier = nn.Linear(1024, 2)
        elif DATASET == "emotion":
            cnn.classifier = nn.Linear(1024, 6)
    elif CNN == "alexnet":
        cnn = torchvision.models.alexnet(pretrained=False)
        cnn.classifier[6] = nn.Linear(4096, 2)
        if DATASET == "gender":
            cnn.classifier[6] = nn.Linear(4096, 2)
        elif DATASET == "emotion":
            cnn.classifier[6] = nn.Linear(4096, 6)
    elif CNN == "mobilenetv3":
        cnn = torchvision.models.mobilenet_v3_small(pretrained=True)
        cnn.classifier[3] = nn.Linear(1024, 6)
    elif CNN == "resnet50":
        cnn = torchvision.models.resnet50(pretrained=True)
        num_ftrs = cnn.fc.in_features
        if DATASET == "gender":
            cnn.fc = nn.Linear(num_ftrs, 2)
        elif DATASET == "emotion":
            cnn.fc = nn.Linear(num_ftrs, 6)

    cnn = cnn.to(device)

    loss_fn = nn.CrossEntropyLoss() 
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)
    step_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)

    # training the model
    train(cnn, train_dataloader, loss_fn, optimiser, device, step_lr_scheduler, EPOCHS, validation_dataloader)
    

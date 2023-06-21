import torch
import seaborn as sns
import pandas as pd
from  sklearn.metrics import confusion_matrix
import torchaudio
from dataset import audioDataset
from train import SAMPLE_RATE, NUM_SAMPLES
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt 
import torchvision 

NUM_SAMPLES = 32000
SAMPLE_RATE = 16000

DATASET = "gender"
CNN = "resnet50"

if DATASET == "gender":
    AUDIO_DIR = "H:\\voxc2\\voxceleb2\\test\\aac" #replace with own path
if DATASET == "emotion":
    AUDIO_DIR = "EMOTION_TEST" # replace with own path

if (DATASET == "gender"):
        class_mapping = [
        "female",
        "male"
    ]
elif (DATASET == "emotion"):
    class_mapping = [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
    ]
def predict(model, input, target, class_mapping):
    model.eval() 
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":

    cuda_ = "cuda:0"
    print(cuda_)

    if torch.cuda.is_available():
        print("cuda available")
        device = "cpu"
    else:
        device = "cpu"
    print(f"Using {device}")


    if DATASET == "gender":
        if CNN == "densenet121":
            cnn = torchvision.models.densenet121(num_classes = 2)
        elif CNN == "alexnet":
            cnn = torchvision.models.alexnet(num_classes = 2)
        elif CNN == "mobilenetv3":
            cnn = torchvision.models.mobilenet_v3_small(num_classes = 2)
        elif CNN == "resnet50":
            cnn = torchvision.models.resnet50(num_classes = 2)
    elif DATASET == "emotion":
        if CNN == "densenet121":
            cnn = torchvision.models.densenet121(num_classes = 6)
        elif CNN == "alexnet":
            cnn = torchvision.models.alexnet(num_classes = 6)
        elif CNN == "mobilenetv3":
            cnn = torchvision.models.mobilenet_v3_small(num_classes = 6)
        elif CNN == "resnet50":
            cnn = torchvision.models.resnet50(num_classes = 6)

    state_dict = torch.load(f"models/{CNN}_{DATASET}_final.pth")

    cnn.load_state_dict(state_dict)
    cnn = cnn.to(device)

    # creating mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
    )

    # creating the custom dataset
    emotionData = audioDataset(
                            AUDIO_DIR,
                            None, 
                            mel_spectrogram,
                            NUM_SAMPLES,
                            SAMPLE_RATE,
                            device,
                            DATASET)
    
    count = 0
    correct = 0
    skipped = 0
    y_true = []
    y_pred = [] 

    while count < len(emotionData):
        input = emotionData[count][0]
        target = emotionData[count][1]
        if target < len(class_mapping):
            input.unsqueeze_(0)
            predicted, expected = predict(cnn, input, target, class_mapping)
            if (predicted == expected):
                correct = correct + 1
            y_pred.append(class_mapping.index(predicted))
            y_true.append(class_mapping.index(expected))
        else:
            skipped = skipped +1
        count = count + 1

    # confusion matrix
    print("acc = ", correct / len(emotionData))
    print(skipped)
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_mapping, digits=4))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(cm)
    print("result predicted ", predicted, " expected ", expected)

    # plots
    ax = plt.subplot()
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, annot=True, ax=ax, fmt="g", cmap='Blues')
    ax.set_xlabel('Predicted class', size=17);
    ax.set_ylabel('Actual class', size=17); 
    ax.set_title(''); 
    if DATASET == "emotion":
        ax.xaxis.set_ticklabels(['Anger', 'Disgust', "Fear", "Happiness", "neutral", "sadness"], size=12);
        ax.yaxis.set_ticklabels(['Anger', 'Disgust', "Fear", "Happiness", "neutral", "sadness"], size=12);
    if DATASET == "gender":
        ax.xaxis.set_ticklabels(['Female', 'Male'], size=12);
        ax.yaxis.set_ticklabels(['Female', 'Male'], size=12);
    plt.show()
# ====================================================================== #
# (1) Didn't overfit a single Batch
#     - 모델 Setting이 끝났다고 바로 학습에 들어가지 말 것!
#     - take out a single batch
#     - train epoch times with single batch -> overfit a single batch
#     - 엄청나게 많이 도와줄거야!! 디버깅하는데 쓰는 시간 낭비를 줄여줄거야!


# (2) Forgot toggle train/eval mode
#     - dropout, batchnorm 같이 model에 따라서 다르게 적용되므로 주의


# (3) Forgot to Zero grad
#     - 매우 큰 차이를 만들어 내기 때문에 꼭 사용해줘야 한다
#     - zero grad를 하지 않으면 gradient가 축적되기 때문에 주의가 필요


# (4) Using softmax with Cross Entropy Loss
#     - cross entropy loss는 softmax를 내장하고 있기 때문에 logit값을 넘겨줘야한다.


# (5) Using bias when using BatchNorm
#     - 특정 neural network layer이후에 BatchNorm을 사용할 경우에 bias를 안써도 된다
#     - BatchNorm 자체에 bias가 있기 때문에 불필요


# (6) Using view as permute
#     - View는 단순히 순서대로 나열되어 있는 원소들을 다시 형태에 맞게 재배열
#     - 따라서 permutation이 필요하다면 view, reshape이 아닌 permute method를 사용


# (7) Using bad data augmentation
#     - 다른 사람들이 쓴 augmentation 기법을 단순히 끌어다와서 사용하는 것이 아니라
#     - 진행하고 있는 task의 output을 고려하고 input의 분포를 고려해서 적절한 augmentation 방법을 찾아야 한다.


# (8) Not shuffling the data
#     - 단, time series data는 주의가 필요하다.


# (9) Not normalizing the data
#     - Data의 mean과 std를 먼저 계산하고 이를 적용


# (10) Not clipping gradients
#     - RNN, GRU, LSTM ...
#     - Exploding gradient problems를 방지하기 위해서 사용

# ===============================

# Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
# batch_size = 64
# num_epochs = 3
batch_size = 2  # check for single batch with small size
num_epochs = 1000  # make overfitting to single batch (loss goes to almost zero)


# bad data augmentation
my_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,1307,), std=(0.3081,))
])


# Load Data
train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    transform=my_transforms,
    download=True
)
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=my_transforms,
    download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Take single batch from train dataloader
data, targets = next(iter(train_loader))

# Train Network
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}")
    # for batch_idx, (data, targets) in enumerate(train_loader): # first check for single batch

    # Get data to cuda if possible
    data = data.to(device)
    targets = targets.to(device)

    # Get to correct shape
    data = data.reshape(data.shape[0], -1)

    # forward
    scores = model(data)
    loss = criterion(scores, targets)

    # backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

    # gradient descent
    optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(
            f"Got {num_correct} / {num_samples}"
            f"with accuracy {float(num_correct) / float(num_samples) * 100: .2f}"
        )

    model.train()

check_accuracy(test_loader, model)
import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets

## 1. Class weighting
loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, 50]))


## 2. Oversampling
def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    # class_weights = [1, 50] # class의 분포에 따라서 weight 지정 (반비례)

    # generalize calculating class weights
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    sample_weights = [0] * len(dataset) # 각 data point에 대해서 weight를 지정

    for idx, (data, label) in enumerate(dataset): # 모든 data point를 돌면서
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight # 각 data point에 지정된 class_weight를 제공
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # replacement: If false, they are drawn without replacement,
    #              which means that when a sample index is drawn for a row, it cannot be drawn again for that row.
    #              (즉, false로 지정하면 한번만 데이터를 본다는 의미이므로 oversampling이 안됨)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler) # dataloader의 sampler로 지정
    return loader


def main():
    loader = get_loader(root_dir="dataset", batch_size=8)

    for data, labels in loader:
        print(labels)

if __name__ == '__main__':
    main()
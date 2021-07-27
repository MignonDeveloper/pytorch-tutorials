import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    """Some Information about VOCDataset"""
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # text -> clss, x, y, w, h(midpoint format)
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                clas_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # ground truth는 박스 하나당 하나의 ground truth를 가진다.

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i,j = int(self.S * y), int(self.S * x) # 어떤 cell에 속하는지 확인 -> ground truth x,y는 0~1 사이의 값으로 주어진다.
            x_cell, y_cell = self.S * x - j, self.S * y - i # cell안에서 midpoint가 어디에 있는지 계산
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return label_matrix

    def __len__(self):
        return len(self.annotations)
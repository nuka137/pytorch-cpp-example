import os
import numpy as np

import cv2
import torch
import torchvision


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_list_text):
        with open(image_list_text, "r") as f:
            image_list = f.readlines()
        data = []
        for image in image_list:
            d = {}
            d["image_path"] = image.rstrip("\n")
            d["id"] = os.path.splitext(os.path.basename(image.rstrip("\n")))[0]
            d["label_path"] = image.rstrip("\n").replace("JPEGImages", "labels").replace(".jpg", ".txt")
            data.append(d)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]["image_path"]
        label_path = self.data[index]["label_path"]

        # Load image.
        image = cv2.imread(image_path)
        image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        image = (image - np.array(mean_rgb, dtype=np.float32)) / 255.0
        image = torchvision.transforms.ToTensor()(image)

        # Load label.

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            label_list = f.readlines()
        for label in label_list:
            label = label.strip()
            sp = label.split()
            c = int(sp[0])
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            boxes.append([x1, y1, x2, y2])
            labels.append(c)

        return image, boxes


def test():
    dataset = VOCDataset("data/train.txt")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    count = 0
    for batch_idx, (data, target) in enumerate(loader):
        print(target)

        count += 1
        if count == 10:
            break


if __name__ == "__main__":
    test()

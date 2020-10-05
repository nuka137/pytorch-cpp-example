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
        image_width, image_height, _ = image.shape
        image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mean_rgb = [122.67891434, 116.66876762, 104.00698793]
        image = (image - np.array(mean_rgb, dtype=np.float32)) / 255.0
        image = torchvision.transforms.ToTensor()(image)

        # Load bounding box and label.
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            label_list = f.readlines()
        for label in label_list:
            label = label.strip()
            sp = label.split()
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            c = int(sp[0])
            boxes.append([x1, y1, x2, y2])
            labels.append(c)
        boxes = torch.Tensor(boxes)
        labels = torch.LongTensor(labels)

        boxes /= torch.Tensor([[image_width, image_height, image_width, image_height]]).expand_as(boxes)

        S = 7
        B = 2
        C = 20
        N = 5 * B + C       # [x, y, w, h, confidence] * num_boxes + num_classes

        boxes_wh = boxes[:, 2:] - boxes[:, :2]
        boxes_xy = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        target = torch.zeros([S, S, 5 * B + C])
        cell_size = 1.0 / float(S)
        for b in range(boxes.size(0)):
            xy = boxes_xy[b]
            wh = boxes_wh[b]
            label = labels[b]
            
            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])
            x0y0 = ij * cell_size
            xy_normalized = (xy - x0y0) / cell_size

            for k in range(B):
                s = 5 * k
                target[j, i, s:s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4] = 1.0
            target[j, i, 5*B + label] = 1.0

        return image, target


def test():
    dataset = VOCDataset("data/train.txt")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    count = 0
    for batch_idx, (data, target) in enumerate(loader):
        print(target.shape)

        count += 1
        if count == 10:
            break


if __name__ == "__main__":
    test()

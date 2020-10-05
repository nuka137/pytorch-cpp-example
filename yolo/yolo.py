import torch
import torch.nn.functional as F
from torchsummary import summary

from voc_loader import VOCDataset

NUMBER_OF_EPOCHS = 10
LOG_INTERVAL = 100
#BATCH_SIZE = 64
BATCH_SIZE = 1


def compute_iou(bbox1, bbox2):
    # bbox1: [N, 4(x1, y1, x2, y2)]
    # bbox2: [M, 4(x1, y1, x2, y2)]
    N = bbox1.size(0)
    M = bbox2.size(0)

    lt = torch.max(
        bbox1[:, :2].unsqueeze(1).expand(N, M, 2),
        bbox2[:, :2].unsqueeze(0).expand(N, M, 2)
    )
    rb = torch.min(
        bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),
        bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)
    )

    wh = rb - lt
    wh[wh < 0] =0

    area_inter = wh[:, :, 0] * wh[:, :, 1]
    area_bbox1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area_bbox2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    area_bbox1 = area_bbox1.unsqueeze(1).expand_as(area_inter)
    area_bbox2 = area_bbox2.unsqueeze(0).expand_as(area_inter)
    area_union = area_bbox1 + area_bbox2 - area_inter

    iou = area_inter / area_union

    return iou



class YoloLoss(torch.nn.Module):
    def __init__(self, num_features=7, num_boxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()

        self.S = num_features
        self.B = num_boxes
        self.c = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predict, target):
        S, B, c = self.S, self.B, self.c
        N = 5 * B + c       # [x, y, w, h, confidence] * num_boxes + num_classes

        coord_mask = target[:, :, :, 4] > 0
        noobj_mask = target[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        print(predict.shape)
        print(target.shape)

        coord_pred = predict[coord_mask].view(-1, N)        # [num_coords, [x, y, w, h, confidence] * num_boxes + num_classes]
        bbox_pred = coord_pred[:, :5*B].view(-1, 5)         # [num_coords * num_boxes, [x, y, w, h, confidence]]
        class_pred = coord_pred[:, 5*B:]                    # [num_coords * num_boxes, num_classes]

        coord_target = target[coord_mask].view(-1, N)       # [num_coords, [x, y, w, h, confidence] * num_boxes + num_classes]
        bbox_target = coord_target[:, :5*B].view(-1, 5)     # [num_coords * num_boxes, [x, y, w, h, confidence]]
        class_target = corrd_target[:, 5*B:]                # [num_coords * num_boxes, num_classes]

        noobj_pred = predict[noobj_mask].view(-1, N)        # [num_coords, [x, y, w, h, confidence] * num_boxes + num_classes]
        noobj_target = target[noobj_mask].view(-1, N)       # [num_coords * num_boxes, num_classes]

        # Compute loss for noobj.
        noobj_conf_mask = torch.zeros(noobj_pred.shape)     # [num_coords, [x, y, w, h, confidence] * num_boxes + num_classes]
        for i in range(B):
            noobj_conf_mask[:, 4 + 5 * B] = 1 
        noobj_pred_conf = noobj_pred[noobj_conf_mask]       # [num_coords, [confidence]]
        noobj_target_conf = noobj_prd[noobj_conf_mask]      # [num_coords, [confidence]]
        loss_noobj_conf = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction="sum")


        # Compute loss for coord.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B]                                         # [num_boxes, [x, y, w, h, confidence]]
            pred_xyxy = Variable(torch.FloatTensor(pred.size()))            # [num_boxes, [x, y, w, h, confidence]]
            pred_xyxy[:, :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i].view(-1, 5)                             # [1, [x, y, w, h, confidence]]
            target_xyxy = Variable(torch.FloatTensor(target.size()))        # [1, [x, y, w, h, confidence]]
            target_xyxy[:, :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            
            coord_response_mask[i+mask_index] = 1

        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)         # [n_response, [x, y, w, h, confidence]]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)     # [n_response, [x, y, w, h, confidence]]
        # TODO: bbox_target_iou
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)           # [n_response, [x, y, w, h, confidence]]
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction="sum")
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction="sum")
        loss_conf = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction="sum")

        loss_class = F.mse_loss(class_pred, class_target, reduction="sum")

        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_conf + self.lambda_noobj * loss_noobj_conf + loss_class
        loss /= predict.size(0)

        return loss


class YoloV1(torch.nn.Module):
    def __init__(self):
        super(YoloV1, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(192, 128, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

        S = 7
        B = 2
        C = 20

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7*7*1024, 4096),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, S * S * (5 * B + C)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # TODO: add pretrained model.
        out = self.conv_layers(x)
        out = self.fc_layers(out)

        return out


def set_random_seed(seed):
    torch.manual_seed(seed)


def get_learning_rate(epoch, iteration, iteration_max):
    if epoch == 0:
        lr = 0.001 + (0.01 - 0.001) * iteration / iteration_max
    elif 1 <= epoch < 76:
        lr = 0.01
    elif 76 <= epoch < 96:
        lr = 0.001
    else:
        lr = 0.0001
    
    return lr


def main():
    set_random_seed(1)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Train on GPU.")
    else:
        device_type = "cpu"
        print("Train on CPU.")
    device = torch.device(device_type)

    # Build model.
    model = YoloV1()
    model.to(device)
    summary(model, (3, 448, 448))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    yolo_loss = YoloLoss()

    # Load dataset.
    train_dataset = VOCDataset("data/train.txt")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    iteration_max = len(train_loader)
    print(f"======== {iteration_max}")

    val_dataset = VOCDataset("data/test.txt")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Train loop.
    for epoch in range(NUMBER_OF_EPOCHS):
        print("Epoch {}: ".format(epoch))

        # Train.
        print("Start train.")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Update learning rate.
            for group in optimizer.param_groups:
                group["lr"] = get_learning_rate(epoch, batch_idx, iteration_max)

            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            pred = model(data)

            loss = yolo_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % LOG_INTERVAL == 0:
                print("Batch: {}, Loss: {}".format(batch_idx, loss.item()))
        
        # Evaluate.
        print("Start eval.")
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                pred = model(data)

                loss = yolo_loss(pred, target)
                

if __name__ == "__main__":
    main()

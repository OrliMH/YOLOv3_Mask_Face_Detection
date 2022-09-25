"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim
import os 

from model import YOLOv3
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from dataset import YOLODataset
from terminaltables import AsciiTable
from utils import (
    load_checkpoint,
    get_loaders,
    xywh2xyxy,
    non_max_suppression_kkb,
    get_batch_statistics,
    ap_per_class,
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer):
    losses = []
    mean_loss = None 
    for batch_idx, (_, x, y) in enumerate(tqdm(train_loader, desc="train_process")):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        # y: b nbboxes 6


        loss, out = model(x, y) # loss, yolo_outputs loss:sum of layer_losses yolo_outputs:3*3*S*S 4+1+cls
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
    return mean_loss

def evaluate(model, valid_loader, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm(valid_loader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            loss, outputs = model(imgs) # b 3*3*S*S 4+1+cls
            # print(outputs.shape) # torch.Size([8, 10647, 7])
            outputs = non_max_suppression_kkb(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    train_loader, valid_loader = get_loaders(
        train_txt_path=config.TRAIN_TXT_PTH, test_txt_path=config.TEST_TXT_PTH
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    max_mAP = -1
    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        loss = train_fn(train_loader, model, optimizer)

        if epoch > 0 and epoch % 3 == 0:
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                valid_loader,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=config.IMAGE_SIZE,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            print(evaluation_metrics)
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, config.MASK_CLASSES[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            if epoch > config.NUM_EPOCHS//3:
                cur_mAP = AP.mean()
                if cur_mAP > max_mAP:
                    max_mAP = cur_mAP
                    checkpoint_file = "global_max_mAP_"+str(max_mAP)+".pth.tar"
                    checkpoint_pth = os.path.join(config.CHECKPOINT_DIR, checkpoint_file)
                    torch.save(model.state_dict(), checkpoint_pth)
                    print("save model dict !!!")
            model.train()


if __name__ == "__main__":
    main()

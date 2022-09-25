"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union, build_targets, to_cpu


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()



        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

        self.ignore_thresh = 0.5

    def forward(self, predictions, target, anchors):

        # anchors: scaled anchors (3, 2) 

        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

   
        # iou_scores: [b, num_anchor, grid_size, grid_size] -> pred_boxes与ground_truth的IoU
        # class_mask: [b, num_anchor, grid_size, grid_size], 预测正确的class 为true
        # obj_mask : [b, num_anchor, grid_size, grid_size] -> 1: 一定是正样本落在的地方(b_id, anchor_id, i, j)
        #                                                  -> 0: 一定不是正样本落在的地方
        # noobj_mask:  [b, num_anchor, grid_size, grid_size] -> 1: 一定是负样本落在的地方
        #                                                    -> 0: 不一定是正样本落在的地方，也可能是不参与计算
        #                                                          体现了ignore_thres的价值。>ignore的，都不参与计算
        # 底下是，算出来的，要参与产生loss的真实target.(除了tcls)
        # The procedure to generate those t·, corresponding to the gray circle in slides, can be called as Encode
        # tx: [b, num_anchor, grid_size, grid_size]
        # ty: [b, num_anchor, grid_size, grid_size]
        # tw: [b, num_anchor, grid_size, grid_size]
        # th: [b, num_anchor, grid_size, grid_size]
        # tcls :[b, num_anchor, grid_size, grid_size, n_classes]
        #

        # prdictions  b 3*S*S 4+1+cls
        batch_size = predictions.size(0)
        S = int(torch.sqrt(predictions.size(1)//3))
        predictions = predictions.view(batch_size, 3, S, S, -1)
        pred_boxes = predictions[..., :4]
        pred_cls = predictions[..., 5:]

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,              # (b, 3, 13, 13, 4)
            pred_cls=pred_cls,                  # (b, 3, 13, 13, 80)
            target=target,                     # (n_boxes, 6) [details in build_targets function]
            anchors=anchors,        # (3, 2) 3个anchor，每个2维
            ignore_thres=self.ignore_thresh,     # 0.5 (hard code in YOLOLayer self.init()) anchor_iou 大于这个值，即使它不是obj，也不算noobj 
        )

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        # 可以看到，真正参与loss计算的，仍然是·与t·，即offset regress
        # Reg Loss
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

        # Conf Loss
        # 因为这里conf选择的是bce_loss，因为对于noobj，基本都能预测对，所以loss_conf_noobj通常比较小
        # 所以此时为了平衡，noobj_scale往往大于obj_scale, (100, 1)
        # 实际上，这里的conf loss就是做了个0-1分类，0就是noobj, 1就是obj
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

        # Class Loss
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        # Total Loss
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()     # class_mask/obj_mask(b, 3, 13, 13) # 正确率
        conf_obj = pred_conf[obj_mask].mean()           # 有物体的平均置信度
        conf_noobj = pred_conf[noobj_mask].mean()       # 无物体的平均置信度
        conf50 = (pred_conf > 0.5).float()              # 置信度大于0.5的位置 (b, num_anchor, 13, 13)
        iou50 = (iou_scores > 0.5).float()              # iou大于0.5的位置 (b, num_anchor, 13, 13)
        iou75 = (iou_scores > 0.75).float()             # iou大于0.75的位置 (b, num_anchor, 13, 13)
        detected_mask = conf50 * class_mask * tconf     # tconf=obj_mask, 即：既是预测的置信度>0.5，又class也对，又是obj
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss": to_cpu(total_loss).item(),
            "x": to_cpu(loss_x).item(),
            "y": to_cpu(loss_y).item(),
            "w": to_cpu(loss_w).item(),
            "h": to_cpu(loss_h).item(),
            "conf": to_cpu(loss_conf).item(),
            "cls": to_cpu(loss_cls).item(),
            "cls_acc": to_cpu(cls_acc).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
            "grid_size": grid_size,
        }

        return output, total_loss


        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

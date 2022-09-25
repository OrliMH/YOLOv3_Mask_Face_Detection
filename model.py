"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn
from utils import build_targets, to_cpu
from config import ANCHORS, S

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, image_dim=416):
        super().__init__()
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()        # binary cross entropy
        self.obj_scale = 1                  # lambda们
        self.noobj_scale = 100
        self.metrics = {}                   # 一堆计算变量
        self.num_anchors = 3
        self.image_dim = image_dim 
        self.all_anchor = ANCHORS # (0, 1)

        
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes
    def compute_grid_offsets(self, grid_size, cuda=True):
        # 0<-13; 13<-26; 26<-52
        self.grid_size = grid_size
        g = self.grid_size          # 13, 26, 52
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.image_dim / self.grid_size     # 32, 16, 8 => pixels per grid/feature point represents
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g): tensor([0,1,2,...,12])
        # torch.arange(g).repeat(g, 1): # torch.arange(g).repeat(13, 1) torch.arange(g)这个变量dim0重复13次， dim1重复1次
        #       tensor([[0,1,2,...,12],
        #               [0,1,2,...,12],
        #               ...
        #               [0,1,2,...,12]])
        #       shape=torch.Size([13, 13])
        # torch.arange(g).repeat(g, 1).view([1, 1, g, g]):
        #       tensor([[[[0,1,2,...,12],
        #                 [0,1,2,...,12],
        #                 ...
        #                 [0,1,2,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        # todo: 关于 repeat (不是todo，就是为了这个颜色)
        # torch.repeat(m): 在第0维重复m次
        #                  此处如果只用.repeat(g),则会出现[0,1,...,12,0,1,...12,...,0,1,...12]
        # torch.repeat(m, n): 在第0维重复m次，在第1维重复n次

        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]):
        #       tensor([[[[0,0,0,...,0],
        #                 [1,1,1,...,1],
        #                 ...
        #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])
        # self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]) # [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]*13
        # FloatTensor()后会将里面的tuple()变成[]
        # 将anchor变到(0, 13)范围内
        # self.scaled_anchors = tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]) # 3x2
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        # self.scaled_anchors[:, :1]: tensor([[3.625], [4.8750], [11.6562]])
        # self.anchor_w =
        # self.scaled_anchors.view((1, 3, 1, 1)) =
        #                                          tensor([
        #                                                  [
        #                                                    [[3.625]],
        #                                                    [[4.8750]],
        #                                                    [[11.6562]]
        #                                                  ]
        #                                                 ])
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, cuda=True):
        # x.shape:torch.Size([32, 512, 13, 13]) # b c h w 
        x = self.pred(x)
        # b (num_classes + 5) * 3 h w



        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        # print("x.shape:{}".format(x.shape))
        num_samples = x.size(0)     # batch size
        grid_size = x.size(2)       # feature map size: 13, 26, 52  # initially, self.grid_size = 0
        if grid_size == 13:
            self.scaled_anchors = FloatTensor(self.all_anchor[0])*S[0] # [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]]*13
        elif grid_size == 26:
            self.scaled_anchors = FloatTensor(self.all_anchor[1])*S[1]
        else:
            self.scaled_anchors = FloatTensor(self.all_anchor[2])*S[2]
        # print("self.scaled_anchors:{}".format(self.scaled_anchors))
        # RuntimeError: shape '[32, 3, 7, 13, 13]' is invalid for input of size 2768896
        prediction = (
            #       b, 3, 85, 13, 13
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            #       b, 3, 13, 13, 85
            .permute(0, 1, 3, 4, 2) # b num_anchors h w nb_classes+5 || x y w h conf cls
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])  # Center x   # (b,3,13,13)            # 1 +
        y = torch.sigmoid(prediction[..., 1])  # Center y   # (b,3,13,13)            # 1 +
        w = prediction[..., 2]  # Width                     # (b,3,13,13)            # 1 +
        h = prediction[..., 3]  # Height                    # (b,3,13,13)            # 1 +
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf (b,3,13,13)            # 1 + = 5 +
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. (b,3,13,13,80)    # 80 = 85

        # Initially, self.grid_size = 0 != 13, then 13 != 26, then 26 != 52
        # Each time, if former grid size does not match current one, we need to compute new offsets
        # 作用：
        # 1. 针对不同size的feature map (13x13, 26x26, 52x52), 求出不同grid的左上角坐标
        # 2. 将(0, 416)范围的anchor scale到(0, 13)的范围
        #
        self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        # self.grid_x:                             # self.grid_y:
        #       tensor([[[[0,1,2,...,12],          #       tensor([[[[0,0,0,...,0],
        #                 [0,1,2,...,12],          #                 [1,1,1,...,1],
        #                 ...                      #                 ...
        #                 [0,1,2,...,12]]]])       #                 [12,12,12,...,12]]]])
        #       shape=torch.Size([1, 1, 13, 13])   #       shape=torch.Size([1, 1, 13, 13])
        #                                          #
        # self.anchor_w: shape([1, 3, 1, 1])       # self.anchor_h: shape([1, 3, 1, 1])
        # tensor([                                 # tensor([
        #         [                                #         [
        #           [[3.625]],                     #           [[2.8125]],
        #           [[4.8750]],                    #           [[6.1875]],
        #           [[11.6562]]                    #           [[10.1875]]
        #         ]                                #         ]
        #        ])                                #        ])

        # Add offset and scale with anchors
        # 请回想/对照slides中的等式，是目前绝大部分靠回归offset的方法通行的策略
        # x, y, w, h即上文中prediction, 对应t·,也即offset们, 此部分是直接由网络predict出来的, xy经过sigmoid强制到(0,1)
        # grid_xy是grid的左上角坐标[0,1,...,12],
        # 所以xy+grid_xy就是将pred结果(即物体中心点, slides中蓝色bx, by的部分)分布到每个grid中去，(0, 13)
        #
        # 对于wh，由于prediction的结果直接是log()后的(如果忘记，请回看slides：同样也对应蓝色bw,bh的部分)，所以此处要exp
        #
        # 此时，所有pred_boxes都是（0,13）范围的
        # These preds are final outpus for test/inference which corresponds to the blue circle in slides
        # This procedure could also be called as Decode
        #
        # 通常情况下，单纯的preds并不参与loss的计算，而只是作为最终的输出存在，
        # 但是这里依然计算，并在build_targets函数中出现，其目的，在于协助产生mask
        pred_boxes = FloatTensor(prediction[..., :4].shape)     # (b, 3, 13, 13, 4)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (   # * stride(=32对于13x13)，目的是将(0, 13)的bbox恢复到(0, 416)
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        ) # b 3*S*S 4+1+cls
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                    pred_boxes=pred_boxes,              # (b, 3, 13, 13, 4)
                    pred_cls=pred_cls,                  # (b, 3, 13, 13, 80)
                    target=targets,                     # (n_boxes, 6) [details in build_targets function]
                    anchors=self.scaled_anchors,        # (3, 2) 3个anchor，每个2维
                    ignore_thres=self.ignore_thres,     # 0.5 (hard code in YOLOLayer self.init())
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
        
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x, targets=None):
        loss = 0
        yolo_outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                layer_output, layer_loss = layer(x, targets) # layer_output:b 3*S*S 4+1+cls
                yolo_outputs.append(layer_output)
                loss += layer_loss 
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        # outputs = FloatTensor(outputs)
        # out, loss = outputs[:, 0], torch.sum(outputs[:, 1])
        # return out, loss
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1)) # b 3*3*S*S 4+1+cls


        return loss, yolo_outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x) # 3 b 3 S S nbcls+5
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")

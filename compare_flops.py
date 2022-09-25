import torch
from model import YOLOv3
import config 
import os 
import time 
from thop import profile

def compute_flops(model):
    # Create a network and a corresponding input
    inp = torch.rand(1, 3, 416, 416).cuda()

    # Count the number of FLOPs

    flops, params = profile(model, inputs=(inp, ))
    print("flops:{}".format(flops))
    print("params:{}".format(params))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')





def metric_compare(model):    
    # 对比前向推理时间
    random_input = torch.rand((1, 3, 416, 416)).cuda() 
    def obtain_avg_forward_time(input, model, repeat=200):
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output
    avg_infer_time, _ = obtain_avg_forward_time(random_input, model)
    print("avg_infer_time:{}".format(avg_infer_time))
    


if __name__ == '__main__':
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    weight_pth = "YOLOv3/checkpoint_baseline/global_max_mAP_0.6290617010621256.pth.tar"
    model.load_state_dict(torch.load(weight_pth))
    model.eval()
    compute_flops(model)
    metric_compare(model)


    # flops:32802245632.0
    # params:61529119.0
    # FLOPs = 32.802245632G
    # Params = 61.529119M
    # avg_infer_time:0.019628634452819826

    
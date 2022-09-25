# YOLOv3_Mask_Face_Detection
A pytorch implemented YOLOv3 mask face detection project.  
Two classes, one is the face without wearing a mask,  
the other is the face wearing a mask.

# Training
nohup python train.py > train.log 

# Result
|size |mAP   |speed(ms)  |params(M)   |FLOPS(G)   |
|---  |---   |---        |---         |---        |
|416  |0.63  |19.6       |61.5        |32.8       |

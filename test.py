import torch 
from model.FasterRCNNVGG import FasterRCNNVGG16
from data.Vocdata import VOCDetectionDataset
import os
from trainer import FasterRCNNTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
dataset=VOCDetectionDataset("/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007")

faster_rcnn=FasterRCNNVGG16()
trainer=FasterRCNNTrainer(faster_rcnn).cuda()

PATH="../pretrained_model/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth"

# load the model

trainer.load(PATH)
print("load the pretrained model")



import torch 
from model.FasterRCNNVGG import FasterRCNNVGG16
from data.Vocdata import VOCDetectionDataset

dataset=VOCDetectionDataset("/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007")
img,_=dataset[0]

model=FasterRCNNVGG16()
model=model.cuda()

roi_cls_locs,roi_scores,rois,roi_indices=model(img[None].cuda())
print(roi_cls_locs.size())
print(roi_scores.size())
print(rois.size())
print(roi_indices.size())





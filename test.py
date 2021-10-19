from data.Vocdata import VOCDetectionDataset
from model.utils.bboxTools import loc2bbox,bbox2loc
import torch
if __name__=="__main__":
    PATH="/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007"
    test=VOCDetectionDataset(PATH,image_set='train')
    print(test[0])
    print(test[6])
    print(test[200])
    bbox=torch.randn(30,4)*100
    loc=torch.randn(30,4)
    print(bbox2loc(bbox,loc).size())


    
from data.Vocdata import VOCDetectionDataset
from model.utils.bboxTools import bboxIou,generate_anchor_base
import torch
if __name__=="__main__":
    bbox_a=torch.randn(100,4)
    bbox_b=torch.randn(10,4)
    print(bboxIou(bbox_a,bbox_b).size())
    print(generate_anchor_base())


    
from torchvision import transforms
import numpy as np
from PIL import Image

"""
The preprocessing pipeline
1.torchvision.transforms.ToTensor
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to
 a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
 if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) 
 or if the numpy.ndarray has dtype = np.uint8
2.transforms.Resize()
3.transforms.Normalize()
"""
def imgPreprocessing(img,min_size=600,max_size=1000): 
    """
    Input: PIL image
    Output: transformed tensor 
    """
    img_numpy=np.array(img)
    H,W,_=img_numpy.shape
    scale1=min_size/min(H,W)
    scale2=max_size/max(H,W)
    scale=min(scale1,scale2)
    # the pipeline's input is narray
    tr=transforms.Compose([
        transforms.ToTensor(), # [0,1]
        transforms.Resize((int(H*scale),int(W*scale))),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return tr(img_numpy)
"""
Bcause we resize our img,so we must resize our groud truth box,too
"""
def bboxResize(bbox,in_size,out_size):
    """
    params:
    bbox: tensor (n,4), [[xmin,ymin,xmax,ymax],[],...[]]
    in_size: (H,W) ,H->y,W->x
    out_size: (H,W)
    """
    bbox=bbox.copy()
    in_H,in_W=in_size
    out_H,out_W=out_size

    y_scale=float(out_H)/in_H
    x_scale=float(out_W)/in_W

    bbox[:,0]=x_scale*bbox[:,0] # xmin
    bbox[:,1]=y_scale*bbox[:,1] # ymin
    bbox[:,2]=x_scale*bbox[:,2] # xmax
    bbox[:,3]=y_scale*bbox[:,3] # ymax

    return bbox


class Transform:
    def __init__(self,min_size,max_size) -> None:
        self.min_size=min_size
        self.max_size=max_size
    def __call__(self,image,target):
        


if __name__=="__main__":
    # test part
    PATH="/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007/JPEGImages/000004.jpg"
    img=Image.open(PATH)
    img_tensor=imgPreprocessing(img)
    print(img_tensor.size())
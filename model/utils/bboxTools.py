import torch
import numpy as np

def loc2bbox(src_bbox,loc):
    """
    Decode bounding boxes from bounding box offsets and scales
    Inputs:
        src_bbox:tensor-> [n,4] [[xmin,ymin,xmax,ymax],...[]]
        loc: tensor-> [n,4] [[]] [[t_x,t_y,t_w,t_h]]

    Calc:
        g_y=p_h * t_y + p_y
        g_x=p_w * t_x + p_x
        g_h=p_h * exp(t_h)
        g_w=p_w * exp(t_w)
    Returns:
        tensor [n,4] [[xmin,ymin,xmax,ymax],...[]]
    """
    
    out_bbox=torch.zeros_like(src_bbox,dtype=src_bbox.dtype)
    src_H=src_bbox[:,3]-src_bbox[:,1]
    src_W=src_bbox[:,2]-src_bbox[:,0]
    src_X,src_Y=src_bbox[:,0],src_bbox[:,1]
    # calc
    out_Y=src_H*loc[:,1]+src_Y
    out_X=src_W*loc[:,0]+src_X
    out_H=src_H*torch.exp(loc[:,3])
    out_W=src_W*torch.exp(loc[:,2])
    # output
    out_bbox[:,0]=out_X
    out_bbox[:,1]=out_Y
    out_bbox[:,2]=out_X+out_W
    out_bbox[:,3]=out_Y+out_H

    return out_bbox

def bbox2loc(src_bbox,dst_bbox):
    """
    Encodes the source and the destination bounding boxes to "loc"
    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Inputs:
        src_bbox:tensor-> [n,4] [[xmin,ymin,xmax,ymax],...[]]
        dst_bbox:tensor-> [n,4] [[xmin,ymin,xmax,ymax],...[]]
    Calculate: T for Target,O for Origin 
        t_x=(T_x-O_x)/O_w
        t_y=(T_y-O_y)/O_h
        t_w=log(T_w/O_w)
        t_h=log(T_h/O_h)
    Outputs:
        loc: tensor-> [n,4] [[]] [[t_x,t_y,t_w,t_h]]
    """
    loc=torch.zeros_like(src_bbox)

    O_H=src_bbox[:,3]-src_bbox[:,1]
    O_W=src_bbox[:,2]-src_bbox[:,0]
    O_X,O_Y=src_bbox[:,0],src_bbox[:,1]
    T_H=dst_bbox[:,3]-dst_bbox[:,1]
    T_W=dst_bbox[:,2]-dst_bbox[:,0]
    T_X,T_Y=src_bbox[:,0],src_bbox[:,1]

    loc[:,0]=(T_X-O_X)/O_W
    loc[:,1]=(T_Y-O_Y)/O_H
    loc[:,2]=torch.log(T_W/O_W)
    loc[:,3]=torch.log(T_H/O_H)

    return loc

def bboxIou(bbox_a,bbox_b):
    """
    Calculate the Intersection of Unions (IoUs)
    IoU is calculated as a ratio of area of the intersection 
    and area of the union

    Inputs:
        bbox_a (tensor) [N,4] [[xmin,ymin,xmax,ymax],...[]]
        bbox_b (tensor) [K,4] [[xmin,ymin,xmax,ymax],...[]]
    Returns:
        [N,K]
    """
    area_a=(bbox_a[:,2]-bbox_a[:,0])*(bbox_a[:,3]-bbox_a[:,1]) # [N,]
    area_b=(bbox_b[:,2]-bbox_b[:,0])*(bbox_b[:,3]-bbox_b[:,1]) # [N,]
    
    # calc intersections
    topLeft=torch.maximum(bbox_a[:,None,:2],bbox_b[:,:2]) #[N,K,2]
    bottomRight=torch.minimum(bbox_a[:,None,2:],bbox_b[:,2:]) #[N,K,2]

    area_i=torch.clamp(torch.prod(bottomRight-topLeft,dim=2),min=0.0) # 处理不相交的情况
    return area_i / (area_a[:,None]+area_b-area_i)
    

def generate_anchor_base(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]):
    """
    Generate anchor base windows by enumerating aspect ratios and scales
    Inputs:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
    Outputs:
        tensor:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(x_{min}, y_{min}, x_{max}, y_{max})` of a bounding box.
    """
    py=base_size/2.
    px=base_size/2.

    anchor_base=torch.zeros((len(ratios)*len(anchor_scales),4),
                            dtype=torch.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h=base_size*anchor_scales[j]*np.sqrt(ratios[i])
            w=base_size*anchor_scales[j]*np.sqrt(1./ratios[i])

            index=i*len(anchor_scales)+j
            anchor_base[index,0]=px-w/2.
            anchor_base[index,1]=py-h/2.
            anchor_base[index,2]=px+w/2.
            anchor_base[index,3]=py+h/2.
    return anchor_base
    
    
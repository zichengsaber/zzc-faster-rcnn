import torch


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

    

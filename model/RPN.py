import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.modules import padding 

from model.utils.bboxTools import generate_anchor_base
from model.utils.creatorTool import ProposalCreator

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.
    
    Args:
        in_channels (int): The channel size of input
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
        proposal_creator_params (dict):
            Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    """
    def __init__(self,
        in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
        anchor_scales=[8, 16, 32], feat_stride=16,
        proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork,self).__init__()

        self.anchor_base=generate_anchor_base(
            anchor_scales=anchor_scales,ratios=ratios
        )
        self.feat_stride=feat_stride
        self.proposal_layer=ProposalCreator(self,**proposal_creator_params)
        n_anchor=self.anchor_base.size()[0]
        # FCN
        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.score=nn.Conv2d(
            in_channels=mid_channels,
            out_channels=n_anchor*2,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.loc=nn.Conv2d(
            in_channels=mid_channels,
            out_channels=n_anchor*4,
            kernel_size=1,
            stride=1,
            padding=0
        )

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
    
    def forward(self,x,img_size,scale=1.):
        """Forward Region Proposal Network
        
        Here are notations
        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images
                [N,C,H,W]
            img_size (tuple of ints): A tuple  (height,width)
                which contains image size after scaling
            scale (float): The amount of scaling done to the input images after
                reading them from files.
        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, tensor, tensor, tensor):
            rpn_loss: Predicted bounding box offsets and scales for anchors
                [N,H*W*A,4]
            rpn_scores: Preidcted  foreground scores for anchors
                [N,H*W*A,2]
            rois : A bounding box array containing coordinates of proposal boxes
                This is a concatenation of bounding box 
                arrays from multiple images in the batch
                [R',4] R_i predicted bounding box from the i'th image
                R_i=\sum_{i=1}^N R_i
            rois_indices : An array containing indices of images to which 
                ROIs correspond to , Its shape is [R,]
            anchor : Coordinates of enumerated shifted anchors
            Its shape is [H*W*A,4]

        """
        n,_,hh,ww = x.size()
        # 生成anchor
        anchor=_enumerate_shifted_anchor(
            self.anchor_base,
            self.feat_stride,
            hh,
            ww
        )
        n_anchor=anchor.size()[0] // (hh*ww)
        # forward path
        h=F.relu(self.conv1(x))
        rpn_locs=self.loc(h) # [N,n_anchor*4,hh,ww]
        rpn_scroces=self.score(h) # [N,n_anchor*2,hh,ww]

        rpn_locs=rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        rpn_scroces=rpn_scroces.permute(0,2,3,1).contiguous()
        
        # forward path
        rpn_softmax_scores=F.softmax(rpn_scroces.view(n,hh,ww,n_anchor,2),dim=4) # [n,hh,ww,n_anchor,2]

        rpn_fg_scores=rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores=rpn_fg_scores.view(n,-1)
        rpn_scroces=rpn_scroces.view(n,-1,2)

        rois=list()
        roi_indices=list()
        for i in range(n):
            roi=self.proposal_layer(
                rpn_locs[i].cpu(),
                rpn_fg_scores[i].cpu(),
                anchor,img_size,
                scale=scale
            )

            batch_index=i*torch.ones((len(roi),),dtype=torch.int64)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois=torch.cat(rois,dim=0)
        roi_indices=torch.cat(roi_indices,dim=0)

        return rpn_locs, rpn_scroces, rois, roi_indices, anchor

           






def _enumerate_shifted_anchor(anchor_base,feat_stride,height,weight):
    # enumerate all shifted anchors:
    # 
    # add A anchors (1,A,4) to 
    # Cell K shifts (K,1,4) to get 
    # shift anchors (K,A,4)
    # Reshape to (K*A,4)  shifted anchors

    shift_y=torch.arange(0,height*feat_stride,feat_stride)
    shift_x=torch.arange(0,weight*feat_stride,feat_stride)
    """
    A bug in Pytorch 1.9.1
    the default indexing of `torch.meshgrid` is 'ij',but in `numpy.meshgrid` is 'xy'
    In Pytorch 1.10.0
    it add a parameter `indexing` to choose 'ij' mode or 'xy' mode
    `torch.meshgrid(x,y,indexing='xy')` is equal to `np.meshgrid(x,y)`
    """
    shift_y,shift_x=torch.meshgrid(shift_x,shift_y)
    shift=torch.stack((shift_x.ravel(),shift_y.ravel()
                      ,shift_x.ravel(),shift_y.ravel()),dim=1)

    A=anchor_base.size()[0]
    K=shift.size()[0]
   
    anchor=anchor_base.reshape((1,A,4)) + \
        shift.reshape((1,K,4)).contiguous().permute(1,0,2) # [K,A,4]
    anchor=anchor.reshape((K*A,4))
    

    return anchor
    
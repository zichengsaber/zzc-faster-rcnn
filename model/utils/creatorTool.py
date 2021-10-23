from numpy.core.numeric import indices
import torch
import numpy as np
from torchvision.ops import nms,clip_boxes_to_image
from model.utils.bboxTools import bbox2loc,loc2bbox,bboxIou

class ProposalTargetCreator(object):
    # 2000个roi里面筛选128个roi
    """
    Assign ground truth bounding boxes to given RoIs.
    The :method:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    Args:
        n_sample(int): The number of sampled regions
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as 
            a foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in 
            [:obj:`neg_iou_thresh_lo`, :obj:`neg_iou_thresh_hi`)
        neg_iou_thresh_lo (float):

    """
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,neg_iou_thresh_lo=0.0
                ):
        
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
    def __call__(self,roi,bbox,label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals
        This function samples total of `self.n_sample` RoIs
        from the combination of : `roi` and `bbox`.
        The RoIs are assigned with the ground truth class labels
        as well as bounding box offsets to match the ground truth 
        bounding boxes.As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations
        * :math:`S` is the total number of sampled RoIs, which equals 
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.
        Inputs:
            roi (tensor):Region of Interests (RoIs) from which we sample
                Its shape is [R',4]
            bbox (tensor): The coordinates of ground truth bounding boxes.
                Its shape is [R,4]
            label (tensor) : Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.
        Returns:
            (tensor,tensor,tensor)
            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.
        """
        
        # for training,reinforce the roi
        roi=torch.cat((roi,bbox),dim=0)

        pos_roi_per_image=np.round(self.n_sample*self.pos_ratio)
        iou=bboxIou(roi,bbox) # [N,R]
        gt_assignment=iou.argmax(dim=1) # [N,]
        max_iou=iou.max(dim=1)[0] # [N,]

        # Offset range of classes from [0,class-1] to [1,class]
        # The label with value 0 is the background

        gt_roi_label=label[gt_assignment]+1 # [N,]

        # select foreground RoIs as those with >= pos_iou_thresh
        pos_index=torch.where(max_iou>=self.pos_iou_thresh)[0]
        pos_roi_per_this_image=int(min(pos_roi_per_image,len(pos_index)))
        if pos_index.size()[0] >0:
            # since pytorch don't have method like np.random.choice
            # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch/59461812
            indices=torch.randperm(len(pos_index))[:pos_roi_per_this_image]
            pos_index=pos_index[indices]
        
        # select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi)
        neg_index=torch.where((max_iou<self.neg_iou_thresh_hi)&
                               (max_iou >= self.neg_iou_thresh_lo))[0]
        
        neg_roi_per_this_image=self.n_sample-pos_roi_per_this_image
        neg_roi_per_this_image=int(min(neg_roi_per_this_image,neg_index.size()[0]))
        if neg_index.size()[0] >0:
            indices=torch.randperm(len(neg_index))[:neg_roi_per_this_image]
            neg_index=neg_index[indices]
        # The indices that we selecting (both positive and negative)
        keep_index=torch.cat((pos_index,neg_index))
        gt_roi_label=gt_roi_label[keep_index]
        # negative label set zero
        gt_roi_label[pos_roi_per_this_image:]=0
        sample_roi=roi[keep_index]

        # compute offsets and scales to match sampled RoIs to the Ground Truths
        gt_roi_loc=bbox2loc(sample_roi,bbox[gt_assignment[keep_index]])
        gt_roi_loc=((gt_roi_loc-torch.tensor(loc_normalize_mean,dtype=torch.float32))
                   / torch.tensor(loc_normalize_std,dtype=torch.float32))
        
        return sample_roi,gt_roi_loc,gt_roi_label

def _get_inside_index(anchor,H,W):
    # Calc indicies of anchors which are located completely
    # inside of the image
    index_inside=torch.where(
        (anchor[:,0]>=0) &
        (anchor[:,1]>=0) &
        (anchor[:,2]<=W) &
        (anchor[:,3]<=H) 
    )[0]

    return index_inside

class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
        
    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN 

    Offsets and scales to match anchors to the ground truth

    Args:
        n_sample (int): The number of regions to produce
        pos_iou_thresh(float): Anchors with IOU above this 
            threshold will be assigned as postive.
        neg_iou_thresh(float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample=n_sample
        self.pos_iou_thresh=pos_iou_thresh
        self.neg_iou_thresh=neg_iou_thresh
        self.pos_ratio=pos_ratio
    
    def __call__(self,bbox,anchor,img_size):
        """Assign ground truth supervision to sampled subset of anchors

        Types of input arrays and output arrays  are same

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math `(R,4)`
            anchor (array): Coordinates of anchors. Its shape is 
                :math `(S,4)`
            img_size (tuple of int): A tuple : obj:`H,W`,which is 
                a tuple of height and width of an image
        Rerurns:
            (tensor,tensor)
        """
        img_H,img_w=img_size

        n_anchor=len(anchor)
        inside_index=_get_inside_index(anchor,img_H,img_w)
        anchor=anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc=bbox2loc(anchor,bbox[argmax_ious])

        # map up to original set of anchors
        label=self._unmap(label,n_anchor,inside_index,fill=-1)
        loc=self._unmap(loc,n_anchor,inside_index,fill=0)
        
        return loc,label
        
    def _create_label(self,inside_index,anchor,bbox):
        # label: 1 is positive,0 is negative,-1 is don't care
        label=torch.empty((len(inside_index),),dtype=torch.int64)
        label.fill_(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels  first
        label[max_ious<self.neg_iou_thresh]=0
        # positive label: for each gt ,anchor which highest iou
        label[gt_argmax_ious]=1
        # positive label : above threshold IOU
        label[max_ious>=self.pos_iou_thresh]=1

        # subsample positive labels if we have too many
        n_pos=int(self.pos_ratio*self.n_sample)
        pos_index=torch.where(label==1)[0]

        if len(pos_index)>n_pos:
            indices=torch.randperm(len(pos_index))[:len(pos_index)-n_pos]
            dis_indices=pos_index[indices]
            label[dis_indices]=-1
        
        # subsample negative labels if we have too many
        n_neg=self.n_sample-torch.sum(label==1).item()
        neg_index=torch.where(label==0)[0]

        if len(neg_index)>n_neg:
            indices=torch.randperm(len(neg_index))[:len(neg_index)-n_neg]
            dis_indices=neg_index[indices]
            label[dis_indices]=-1
        
        return argmax_ious,label


    # cannot understand the function
    def _unmap(self,data,count,index,fill=0):
        if len(data.size())==1:
            ret=torch.empty((count,),dtype=data.dtype)
            ret.fill_(fill)
            ret[index]=data
        else:
            ret=torch.empty((count,)+data.shape[1:],dtype=data.dtype)
            ret.fill_(fill)
            ret[index,:]=data
        return ret 


    
    def _calc_ious(self,anchor,bbox,inside_index):
        # ious between the anchors and the gt boxes
        ious=bboxIou(anchor,bbox) #[N,R]
        argmax_ious=ious.argmax(dim=1) #[N,] 对应的bbox的index
        max_ious=ious.max(dim=1)[0] #[N,]
        gt_argmax_ious=ious.argmax(dim=0) #[R,]
        gt_max_ious=ious.max(dim=0)[0] #[R,]

        return argmax_ious,max_ious,gt_argmax_ious


class ProposalCreator:
    """Proposal regions are generated by calling this object
    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.
    
    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.


    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    
    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    
    def __call__(self,loc,score,anchor,img_size,scale=1.0):
        """Inputs should be tensor
        Propose RoIs

        Inputs: obj:`loc ,score,anchor` refer to the same anchor
        by the same index

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Args:
            loc (tensor): Predicted offsets and scaling to anchors.
                [R,4]
            score (tensor): Predicted foreground probability for anchors
                [R,]
            anchor (tensor): Coordinates of anchors. Its shape is 
                [R,4]
            img_size (tuple of ints): (H,W)
            scale (float): The scaling factor used to scale an image after reading
            it from a file
        Returns:
            (tensor):
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.
        """

        if self.parent_model.training:
            n_pre_nms=self.n_train_pre_nms
            n_post_nms=self.n_train_post_nms
        else:
            n_pre_nms=self.n_test_pre_nms
            n_post_nms=self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations
        roi=loc2bbox(anchor,loc) # [R,4] [xmin,ymin,xmax,ymax]
        # clip predited boxes to images
        # using torchvision.ops.clip_boxes_to_image
        roi=clip_boxes_to_image(roi,img_size)

        # Remove predicted boxes with either height or width < threshold
        min_size=self.min_size*scale
        hs=roi[:,3]-roi[:,1]
        ws=roi[:,2]-roi[:,0]
        keep=torch.where((hs>=min_size) & (ws>=min_size))[0]
        roi=roi[keep,:]
        score=score[keep]

        # Sort all (proposal,score) pairs by score from high to
        # low Take top pre_nms_topN
        order=score.ravel().argsort(descending=True)
        if n_pre_nms>0:
            order=order[:n_pre_nms]
        roi=roi[order,:]
        score=score[order]

        # Apply nms (e.g. threshold=0.7)
        # torchvision.ops.nms 's return is indices
        keep=nms(
            roi.cuda(),
            score.cuda(),
            self.nms_thresh
        )

        if n_post_nms > 0:
            keep=keep[:n_post_nms]
        roi=roi[keep.cpu()]

        return roi
        
        
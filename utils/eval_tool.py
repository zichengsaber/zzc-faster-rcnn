import torch
from collections import Counter

def mean_average_precision(pred_bboxes,true_boxes,iou_threshold,num_classes=20):
    
    #pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    #true_boxes:[[]]
    average_precisions=[]#Store AP for each category
    epsilon=1e-6#Prevent denominator from being 0
    
    #For each category
    for c in range(num_classes):
        detections=[]#Store bbox predicted for this category
        ground_truths=[]#Storage itself is bbox(GT) of this category
        
        for detection in pred_bboxes:
            if detection[1]==c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)
                
        #img 0 has 3 bboxes
        #img 1 has 5 bboxes
        #It's like this: amount_bboxes={0:3,1:5}
        #Count the number of real boxes in each picture_ IDX indicates the number of the image to distinguish each image
        amount_bboxes=Counter(gt[0] for gt in ground_truths)
        
        # remember the gt boxes we have covered so far 
        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)#Set to 0, which means that none of these real boxes initially matches any prediction box
        #At this point, amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        
        #Rank the prediction boxes in order of confidence
        detections.sort(key=lambda x:x[2],reverse=True)
        
        #Initialize TP,FP
        TP=torch.zeros(len(detections))
        FP=torch.zeros(len(detections))
        
        #TP+FN is the total number of GT boxes in the current category, which is fixed
        total_true_bboxes=len(ground_truths)
        
        #If there is no GT box in the current category, you can skip it directly
        if total_true_bboxes == 0:
            continue
        
        #For each prediction box, first find all the real boxes in its picture, and then calculate the IoU between the prediction box and each real box. If the IoU threshold is greater than the IoU threshold and the real box does not match other prediction boxes, the prediction result of the prediction box is set as TP, otherwise it is FP
        for detection_idx,detection in enumerate(detections):
            #In the calculation of IoU, it can only be done in the same picture, not between different pictures
            #The number of the picture has the 0 th dimension
            #So the function of the following code is: find all the real boxes in the picture of the current prediction box detection, and use them to calculate the IoU
            ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detections[0]]
            
            num_gts=len(ground_truth_img)
            # 对于pred_bbox这个一张图片中某个类的一个框进行匹配
            best_iou=0
            for idx,gt in enumerate(ground_truth_img):
                #Calculate the IoU of the current prediction box detection and each real box in its picture
                iou=insert_over_union(torch.tensor(detection[3:]),torch.tensor(gt[3:]))
                if iou >best_iou:
                    best_iou=iou
                    best_gt_idx=idx
            if best_iou>iou_threshold:
                #The detection[0] here is amount_ A key of bboxes, representing the number of the picture, best_gt_idx is the subscript of the real box in the value corresponding to the key
                if amount_bboxes[detection[0]][best_gt_idx]==0:#Only the unoccupied real box can be used, and 0 means unoccupied (occupied: the real box matches a prediction box [both IoU are greater than the set IoU threshold])
                    TP[detection_idx]=1#The prediction box is TP
                    amount_bboxes[detection[0]][best_gt_idx]=1#Mark the real box as used and cannot be used for other prediction boxes. Because a prediction box can only correspond to one real box at most (at most: when IoU is less than IoU threshold, the prediction box has no corresponding real box)
                else:
                    FP[detection_idx]=1#Although the IoU between the prediction box and one of the real boxes is greater than the IoU threshold, the real box has already matched with other prediction boxes, so the prediction box is FP
            else:
                FP[detection_idx]=1#The IoU between the prediction box and each box in the real box is less than the IoU threshold, so the prediction box is directly FP
                
        TP_cumsum=torch.cumsum(TP,dim=0)
        FP_cumsum=torch.cumsum(FP,dim=0)
        
        #apply a formula
        recalls=TP_cumsum/(total_true_bboxes+epsilon)
        precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))
        
        #Add the point [0,1]
        precisions=torch.cat((torch.tensor([1]),precisions))
        recalls=torch.cat((torch.tensor([0]),recalls))
        #Using trapz to calculate AP
        average_precisions.append(torch.trapz(precisions,recalls))
        
    return sum(average_precisions)/len(average_precisions) 


def insert_over_union(boxes_preds,boxes_labels):
    
    box1_x1=boxes_preds[...,0:1]
    box1_y1=boxes_preds[...,1:2]
    box1_x2=boxes_preds[...,2:3]
    box1_y2=boxes_preds[...,3:4]#shape:[N,1]
    
    box2_x1=boxes_labels[...,0:1]
    box2_y1=boxes_labels[...,1:2]
    box2_x2=boxes_labels[...,2:3]
    box2_y2=boxes_labels[...,3:4]
    
    x1=torch.max(box1_x1,box2_x1)
    y1=torch.max(box1_y1,box2_y1)
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    
    
    #Calculate the area of intersection area
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    box1_area=abs((box1_x2-box1_x1)*(box1_y1-box1_y2))
    box2_area=abs((box2_x2-box2_x1)*(box2_y1-box2_y2))
    
    return intersection/(box1_area+box2_area-intersection+1e-6)
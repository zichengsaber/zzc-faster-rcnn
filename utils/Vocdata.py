from torch.utils.data import Dataset, dataset
import os 
import torch
import json,xmltodict
from PIL import Image
from lxml import etree


pascal_voc_classes_dict = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}


class VOCDetectionDataset(Dataset):
    """
    Read and parse PASCALVOC Data set for target detection
    Inherited torch.utils.data.Dataset Class, subclass must override__getitem__
    __len__: Provides the size of the dataset
    __getitem__: Gets the data sample of the given key value
    """
    def __init__(self,voc_root,transforms=None,image_set="train") -> None:
        super().__init__()
        '''
        Parameters:
            voc_root: Dataset root
            transforms: Image preprocessing
            image_set: Select the image set you want to use( train，val，trainval， test)
        '''
        self.root=voc_root # root_path
        self.img_root=os.path.join(self.root,"JPEGImages") # img_path
        self.annotations_root=os.path.join(self.root,"Annotations") # xml_path
        txt_list=os.path.join(self.root,"ImageSets","Main",f"{image_set}.txt")
        self.xml_list=None
        with open(txt_list,'r',encoding='utf8') as file:
            self.xml_list=[item.strip() for item in file.readlines()]
        # category
        self.class_dict=pascal_voc_classes_dict
        # image transforms
        self.transforms=transforms
    
    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        # Read xml file and convert it to dict
        xml_path=os.path.join(self.annotations_root,f"{self.xml_list[index]}.xml")
        with open(xml_path,'r') as fd:
            xml_str=fd.read()
        xml_dict=xmltodict.parse(xml_str)['annotation']

        # read image 
        img_path=os.path.join(self.img_root,xml_dict['filename'])
        image=Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")

        # read each object information in a picture
        boxes=[] # object box
        labels=[] # label
        iscrowd=[] # difficulty
        
        for obj in xml_dict['object']:
            xmin=float(obj['bndbox']['xmin'])
            xmax=float(obj['bndbox']['xmax'])
            ymin=float(obj['bndbox']['ymin'])
            ymax=float(obj['bndbox']['ymax'])
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))
        # change array to tensor
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)
        iscrowd=torch.as_tensor(iscrowd,dtype=torch.int64)
        image_id = torch.tensor([index])
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])

        # build target dictionary
        target={}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd  

        # if preprocessing
        if self.transforms is not None:
            image,target=self.transforms(image,target)
        
        return image,target

if __name__=="__main__":
    PATH="/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007"
    test=VOCDetectionDataset(PATH,image_set='test')
    print(test[0])
    






from utils.Vocdata import VOCDetectionDataset

if __name__=="__main__":
    PATH="/home/ZhangZicheng/ObjectionDetection/VOCdevkit/VOC2007"
    test=VOCDetectionDataset(PATH,image_set='train')
    print(test[0])
    print(test[6])
    print(test[200])
    
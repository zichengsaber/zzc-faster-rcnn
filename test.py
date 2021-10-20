from data.Vocdata import VOCDetectionDataset
from model.utils.creatorTool import ProposalTargetCreator
import torch
if __name__=="__main__":
   # we need better test data
   topleft=torch.rand((2000,2))*150
   btright=topleft+torch.rand((2000,2))*100
   roi=torch.cat((topleft,btright),dim=1)
   topleft=torch.rand((10,2))*150
   btright=topleft+torch.rand((10,2))*100
   bbox=torch.cat((topleft,btright),dim=1)
   label=torch.randint(20,(10,))
   f=ProposalTargetCreator()
   print(f(roi,bbox,label))


    
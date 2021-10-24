from data.Vocdata import VOCDetectionDataset
from model.utils.creatorTool import ProposalCreator, ProposalTargetCreator,AnchorTargetCreator
from model.RPN import RegionProposalNetwork
import torch
if __name__=="__main__":
   x=torch.randn((4,512,32,32))
   net=RegionProposalNetwork()
   print(net(x,(600,800)))

   




    
import argparse
import torch


snapshot1 = '/home/lbycdy/work/crog_lb/model.pth'


param1 =torch.load(snapshot1,map_location=torch.device('cpu'))
if "state_dict" in param1:
    param = param1['state_dict']
for key in param:
    print(key)




import argparse
import torch


snapshot1 = '/home/server-msi/work/CROG/pretrain/rn50-quickgelu-cc12m-f000538c.pt'


param1 =torch.load(snapshot1,map_location=torch.device('cpu'))
for key in param1:
    print(key,param1[key].min(),param1[key].max())

from model.crog import CROG
from loguru import logger
import utils.config as config


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='../config/OCID-VLG/crog_multiple_r50.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg
def build_crog(args):
    model = CROG(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list
args = get_parser()

model, param_list = build_crog(args)

for k, v in model.named_parameters():
    print(k, param1[k].min(), param1[k].max())


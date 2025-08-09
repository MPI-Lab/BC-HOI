import torch.utils.data
import torchvision

from .hico import build as build_hico
from .vcoco import build as build_vcoco
from .caption import build as build_caption

def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
def buile_dataset_caption(image_set,args):
    return build_caption(image_set,args)

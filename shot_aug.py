import random
import torch.nn.functional as F


def shot_aug(args, train_input, train_target, test_input, test_target):
    if args.shot_aug == '':
        return [train_input, train_target, test_input, test_target]
    elif args.shot_aug == 'resize':
        return shot_resize(train_input, train_target, test_input, test_target, args.resos)
    else:
        print('No implementation of shot_aug')


def shot_resize(train_input, train_target, test_input, test_target, resos):
    resolution = resos[random.randint(0, len(resos) - 1)]
    train_input_aug = F.interpolate(train_input, (resolution, resolution), mode='bilinear', align_corners=True)
    test_input_aug = F.interpolate(test_input, (resolution, resolution), mode='bilinear', align_corners=True)
    return [train_input_aug, train_target, test_input_aug, test_target]
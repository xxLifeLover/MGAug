import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
# from torchmeta.datasets import CUB
from cub import CUB
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter
from gbml.maml_higher import MAML
from gbml.fomaml_higher import FOMAML
from mbml.protonet import ProtoNet
from utils import set_seed, set_gpu, check_dir, dict2json, ImageJitter
import time

def train(args, model, dataloader):
    # model.network.train()
    loss_list = []
    acc_list = []
    with tqdm(dataloader, total=args.num_train_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log, _ = model.outer_loop(batch, is_train=True)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_train_batches:
                break
    loss_mean = np.round(np.mean(loss_list), 4)
    loss_err = np.round((1.96 * np.std(loss_list) / np.sqrt(args.num_train_batches)), 2)
    acc_mean = np.round(np.mean(acc_list) * 100, 2)
    acc_err = np.round((1.96 * np.std(acc_list) / np.sqrt(args.num_train_batches)) * 100, 2)
    return loss_mean, loss_err, acc_mean, acc_err


# @torch.no_grad()
def valid(args, model, dataloader):
    # model.network.eval()
    loss_list = []
    acc_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            loss_log, acc_log = model.outer_loop(batch, is_train=False)
            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_valid_batches:
                break
    loss_mean = np.round(np.mean(loss_list), 4)
    loss_err = np.round((1.96 * np.std(loss_list) / np.sqrt(args.num_valid_batches)), 2)
    acc_mean = np.round(np.mean(acc_list) * 100, 2)
    acc_err = np.round((1.96 * np.std(acc_list) / np.sqrt(args.num_valid_batches)) * 100, 2)
    return loss_mean, loss_err, acc_mean, acc_err


def main(args):
    if args.alg=='MAML':
        model = MAML(args)
    elif args.alg=='FOMAML':
        model = FOMAML(args)
    elif args.alg=='ProtoNet':
        model = ProtoNet(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.dataset=='CUB':
        dataclass = CUB
        print("loaded cub")
    else:
        raise ValueError('Not used Data-set ')

    if args.load:
        model.load()
    elif args.load_encoder:
        model.load_encoder()
    train_dataset = dataclass(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='train',
                        transform=transforms.Compose([
                        transforms.RandomResizedCrop(args.image_size),
                        ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225])),
                        ]),
                        download=True,
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    train_loader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)

    valid_dataset = dataclass(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='val',
                        transform=transforms.Compose([
                        transforms.Resize([int(args.image_size * 1.15), int(args.image_size * 1.15)]),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ]),
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    valid_dataset = ClassSplitter(valid_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    valid_loader = BatchMetaDataLoader(valid_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)

    test_dataset = dataclass(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='test',
                        transform=transforms.Compose([
                        transforms.Resize([int(args.image_size * 1.15), int(args.image_size * 1.15)]),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ]),
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)

    print(len(train_loader), len(valid_loader), len(test_loader))

    start_epoch = 0

    if args.resume:
        resume_file = os.path.join(args.save_path, 'last_model.pth')
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.network.load_state_dict(tmp['state'])

    dict2json(os.path.join(args.save_path, 'train_log.json'), {'args': str(args)})

    max_acc = 0.
    for epoch in range(start_epoch, args.num_epoch):
        # init log_dict
        log_dict = {'epoch': epoch}

        # train
        t_start = time.time()
        print('Epoch {}, T {}'.format(epoch, time.strftime("%m-%d %H:%M:%S", time.localtime(t_start))))
        train_loss, train_loss_err, train_acc, train_acc_err = train(args, model, train_loader)
        t_end = time.time()

        # log training states
        log_dict['t_time']     = round((t_end - t_start), 2)
        log_dict['lr']         = round(model.outer_optimizer.state_dict()['param_groups'][0]['lr'], 4)

        log_dict['t_loss']     = train_loss
        log_dict['t_loss_err'] = train_loss_err
        log_dict['t_acc']      = train_acc
        log_dict['t_acc_err']  = train_acc_err
        dict2json(os.path.join(args.save_path, 'train_log.json'), log_dict)
        log_dict['state'] = model.network.state_dict()
        torch.save(log_dict, os.path.join(args.save_path, "last_model.pth"))

        # valid and test
        if epoch % 50 == 0 and epoch != 0 or epoch == args.num_epoch - 1:

            #valid
            v_start = time.time()
            valid_loss, valid_loss_err, valid_acc, valid_acc_err = valid(args, model, valid_loader)
            v_end = time.time()
            log_dict['v_time']     = round((v_end - v_start), 2)
            log_dict['v_loss']     = valid_loss
            log_dict['v_loss_err'] = valid_loss_err
            log_dict['v_acc']      = valid_acc
            log_dict['v_acc_err']  = valid_acc_err

            # save model
            torch.save(log_dict, os.path.join(args.save_path, "epoch" + str(epoch) + ".pth"))
            if valid_acc >= max_acc:
                max_acc = valid_acc
                torch.save(log_dict, os.path.join(args.save_path, "best_model.pth"))

            # test
            test_loss, test_loss_err, test_acc, test_acc_err = valid(args, model, test_loader)
            del log_dict['state']
            log_dict['te_loss']     = test_loss
            log_dict['te_loss_err'] = test_loss_err
            log_dict['te_acc']      = test_acc
            log_dict['te_acc_err']  = test_acc_err
            dict2json(os.path.join(args.save_path, 'val_log.json'), log_dict)

        if args.lr_sched != 'None':
            model.lr_sched()

    return None

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Meta-Learning Algorithms')

    # experimental settings
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='CUB', help='dataset')
    parser.add_argument('--data_path', type=str, default='./data', help='Path of dataset.')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--resume', action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--pin_memory', action='store_false', default=True, help='if or not use pin_memory')

    # training settings
    parser.add_argument('--num_epoch', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_train_batches', type=int, default= 100)
    parser.add_argument('--num_valid_batches', type=int, default= 100)
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=1, help='Number of support examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_query', type=int, default=15, help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--alg', type=str, default='ProtoNet')
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=str, default='None')
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--image_size', type=int, default=84)
    # catfish settings
    parser.add_argument('--net_aug', type=str, default='',
                        choices=['mask-layer-snip-mbml', 'mask-layer-snip-mbml-small', 'mask-layer-snip-fomaml',
                                 'mask-layer-snip-fomaml-small'])
    parser.add_argument('--max_width', type=float, default=1.)
    parser.add_argument('--min_width', type=float, default=0.8)
    parser.add_argument('--num_subnet', type=int, default=3)
    parser.add_argument('--shot_aug', type=str, default='', choices=['resize'])
    parser.add_argument('--resos', type=list, default=[84, 64, 48])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    if 'Conv' in args.net:
        args.image_size = 84
        args.resos = [84, 64, 48]
    else:
        args.image_size = 224
        args.resos = [224, 192, 160, 140]

    save_str = args.alg + '_' + args.dataset + '_' + args.net + '_' + str(args.num_way) + 'w' + str(
        args.num_shot) + 's_' + str(args.net_aug) + '_(' + str(args.min_width) + ',' + str(args.max_width) + ')' + '_' +args.shot_aug
    if args.shot_aug == 'resize':
        save_str = save_str + '_' + str(args.resos)
    args.save_path = os.path.join(args.result_path, save_str)
    check_dir(args.save_path)

    if args.num_epoch == -1:
        if args.num_shot == 1:
            args.num_epoch = 1600
        elif args.num_shot == 5:
            args.num_epoch = 800
        else:
            args.num_epoch = 1600
    if 'maml' in str(args.alg).lower():
        args.batch_size = 4
        args.num_train_batches = 15
        args.num_valid_batches = 25
        if args.num_shot == 1:
            args.num_epoch = 4800
        elif args.num_shot == 5:
            args.num_epoch = 2400
        else:
            args.num_epoch = 4800
    main(args)
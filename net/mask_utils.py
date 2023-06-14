import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


def apply_grad_mask(net, keep_masks):
    prunable_layers = filter(
        lambda layer: is_mask_layer(layer), net.modules())
    handles = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            def hook(grads):
                return grads * keep_mask
            return hook
        handles.append(layer.weight.register_hook(hook_factory(keep_mask)))
    return handles


def apply_weight_mask(net, keep_masks):
    prunable_layers = filter(
        lambda layer: is_mask_layer(layer), net.modules())
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)
        layer.weight_mask = keep_mask
        layer.weight_mask.requires_grad = False
        layer.forward = types.MethodType(snip_forward_conv2d, layer)


def remove_weight_mask(net):
    for i, layer in enumerate(net.modules()):
        if is_mask_layer(layer):
            del layer.weight_mask
            layer.forward = types.MethodType(forward_conv2d, layer)


def remove_grad_mask(handles):
    for handle in handles:
        handle.remove()


def get_mask_list(ratio_list, network, net_aug, train_input, train_target, test_input, test_target):
    if 'snip-mbml' in net_aug:
        score_abs = mask_snip_mbml(network, train_input, train_target, test_input, test_target)
    elif 'snip-fomaml' in net_aug:
        score_abs = mask_snip_fomaml(network)

    if 'layer' in net_aug:
        mask_list = get_mask_layer(score_abs, ratio_list, net_aug)
    else:
        mask_list = get_mask_global(score_abs, ratio_list, net_aug)
    return mask_list


def get_mask_layer(score_abs, ratio_list, mask_type):
    mask_list = [[] for i in ratio_list]
    for score in score_abs:
        layer_scores = torch.flatten(score)
        norm_factor = torch.sum(layer_scores)
        layer_scores.div_(norm_factor)

        # get threshold
        for ii, keep_ratio in enumerate(ratio_list):
            num_params_to_keep = int(len(layer_scores) * ((1 - keep_ratio) if 'small' in mask_type else keep_ratio))
            threshold, _ = torch.topk(layer_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]
            if 'small' in mask_type:
                mask_list[ii].append((score <= acceptable_score).float())
            else:
                mask_list[ii].append((score >= acceptable_score).float())

    return mask_list


def get_mask_global(score_abs, ratio_list, mask_type):
    mask_list = [[] for i in len(ratio_list)]

    all_scores = torch.cat([torch.flatten(x) for x in score_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    for ii, keep_ratio in enumerate(ratio_list):
        num_params_to_keep = int(len(all_scores) * ((1 - keep_ratio) if 'small' in mask_type else keep_ratio))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        for g in score_abs:
            if 'small' in mask_type:
                mask_list[ii].append(((g / norm_factor) <= acceptable_score).float())
            else:
                mask_list[ii].append(((g / norm_factor) >= acceptable_score).float())
    return mask_list

def mask_snip_mbml(net, train_input, train_target, test_input, test_target):
    net_copy = copy.deepcopy(net)
    for layer in net_copy.modules():
        if is_mask_layer(layer):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False
            layer.weight_mask.requires_grad = True
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
    net_copy.zero_grad()
    embedding_support = net_copy(train_input)
    embedding_query = net_copy(test_input)
    prototypes = get_prototypes(embedding_support, train_target)
    test_logit = -get_distance(prototypes, embedding_query, 'euclidean')
    loss = F.cross_entropy(test_logit, test_target)
    loss.backward()

    score_abs = []
    for layer in net_copy.modules():
        if is_mask_layer(layer):
            score_abs.append(torch.abs(layer.weight_mask.grad))
    return score_abs

def mask_snip_fomaml(net):
    score_abs = []
    for layer in net.modules():
        if is_mask_layer(layer):
            score_abs.append(torch.abs(layer.weight.grad * layer.weight.data))
    return score_abs


def is_mask_layer(layer):
    return isinstance(layer, nn.Conv2d)


def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def forward_conv2d(self, x):
    return F.conv2d(x, self.weight, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def get_prototypes(inputs, targets):
    n_way = torch.unique(targets).size(0)
    k_shot = targets.size(0) // n_way
    embed_dim = inputs.size(-1)
    indices = targets.unsqueeze(-1).expand_as(inputs)
    prototypes = inputs.new_zeros(n_way, embed_dim)
    prototypes.scatter_add_(0, indices, inputs).div_(k_shot)
    return prototypes


def get_distance(x, y, distance_type, eps = 1e-10):
    if distance_type == 'euclidean':
        n = x.size(0)
        m = y.size(0)
        x = x.unsqueeze(0).expand(m, n, -1)
        y = y.unsqueeze(1).expand(m, n, -1)
        distance = ((x - y) ** 2).sum(dim=-1)
        return distance
    elif distance_type =='cosine':
        x_norm = F.normalize(x, dim=1, eps=eps)
        y_norm = F.normalize(y, dim=1, eps=eps)
        return (x_norm @ y_norm.t()).t()


def show_gard_hist(count, i, grads, thre):
    if i == 2:
        plt.hist(grads, bins=500)
    elif i == 17:
        plt.hist(grads, bins=5000)
    else:
        plt.hist(grads, bins=5000)
    plt.vlines(thre, 0, plt.gca().get_ylim()[1], color="red", alpha=0.5)
    plt.xlim(plt.gca().get_xlim()[0]*0.1, plt.gca().get_xlim()[1])
    plt.savefig('imgs/layer' + str(count) + str(i) + '-' + str(len(grads)) + '-' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())  + '.png')
    plt.cla()


def check_mask_rate(keep_masks):
    for i, x in enumerate(keep_masks):
        print(torch.count_nonzero(x).item(), '--', x.numel(), "--" ,torch.count_nonzero(x).item() / x.numel() * 100, "%" )


def check_state_0(net):
    for i, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d):
            print('layer-', i, ': ', torch.count_nonzero(layer.weight.data).item(), '-', layer.weight.data.numel(), "-",
                  torch.count_nonzero(layer.weight.data).item() / layer.weight.data.numel() * 100, "%")
            print('mask-', i, ': ', torch.count_nonzero(layer.weight_mask).item(), '-', layer.weight.data.numel(), "-",
                              torch.count_nonzero(layer.weight_mask).item() / layer.weight_mask.numel() * 100, "%")
            print('real-', i, ': ', torch.count_nonzero(layer.weight.data * layer.weight_mask).item(), '-', (layer.weight.data * layer.weight_mask).numel(), "-",
                              torch.count_nonzero(layer.weight.data * layer.weight_mask).item() / (layer.weight.data * layer.weight_mask).numel() * 100, "%")
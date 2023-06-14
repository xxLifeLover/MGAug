import numpy as np
import torch
import torch.nn.functional as F
from net import mask_utils
from mbml.mbml import MBML
from shot_aug import shot_aug
from utils import get_accuracy, AverageMeter


def random_rate_min(min_width=0.9, max_width=1.01, num_subnet=3):
    keep_rate = [min_width]
    keep_rate.extend(list(np.random.uniform(min_width, max_width, num_subnet-1)))
    for k in range(len(keep_rate)):
        if keep_rate[k] > 0.99:
            keep_rate[k] = 0.99
    return keep_rate


class ProtoNet(MBML):
    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()

    def outer_loop(self, batch, is_train):
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)
        loss_log = AverageMeter()
        acc_log = AverageMeter()

        if is_train:
            self.network.zero_grad()
            for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
                embedding_support = self.network(train_input)
                embedding_query = self.network(test_input)
                prototypes = self.get_prototypes(embedding_support, train_target)
                test_logit = -self.get_distance(prototypes, embedding_query, 'euclidean')  # (n_query_samples, n_way)
                outer_loss = F.cross_entropy(test_logit, test_target)
                outer_loss.backward()
                loss_log.update(outer_loss.item())
                with torch.no_grad():
                    acc_log.update(get_accuracy(test_logit, test_target).item())

                if 'mask' in self.args.net_aug:
                    width_mult_list = sorted(
                        random_rate_min(self.args.min_width, self.args.max_width, num_subnet=self.args.num_subnet),
                        reverse=True)
                    keep_mask_list = mask_utils.get_mask_list(width_mult_list, self.network, self.args.net_aug,
                                                              train_input, train_target, test_input, test_target)
                    for keep_mask in keep_mask_list:
                        [train_input_aug, train_target_aug, test_input_aug,
                         test_target_aug] = shot_aug(self.args, train_input, train_target, test_input, test_target)

                        mask_utils.apply_weight_mask(self.network, keep_mask)
                        handle = mask_utils.apply_grad_mask(self.network, keep_mask)
                        embedding_support = self.network(train_input_aug)
                        embedding_query = self.network(test_input_aug)
                        prototypes = self.get_prototypes(embedding_support, train_target_aug)
                        test_logit = -self.get_distance(prototypes, embedding_query, 'euclidean')
                        outer_loss = F.cross_entropy(test_logit, test_target_aug)
                        outer_loss.backward()
                        mask_utils.remove_grad_mask(handle)
                        mask_utils.remove_weight_mask(self.network)
            self.outer_optimizer.step()
            return loss_log.avg, acc_log.avg, 0
        else:
            for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs,
                                                                            test_targets):
                embedding_support = self.network(train_input)
                embedding_query = self.network(test_input)
                prototypes = self.get_prototypes(embedding_support, train_target)
                test_logit = -self.get_distance(prototypes, embedding_query, 'euclidean')
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_log.update(outer_loss.item())
                with torch.no_grad():
                    acc_log.update(get_accuracy(test_logit, test_target).item())
            return loss_log.avg, acc_log.avg

    def get_prototypes(self, inputs: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:

        n_way = torch.unique(targets).size(0)
        k_shot = targets.size(0) // n_way
        embed_dim = inputs.size(-1)
        indices = targets.unsqueeze(-1).expand_as(inputs)
        prototypes = inputs.new_zeros(n_way, embed_dim)
        prototypes.scatter_add_(0, indices, inputs).div_(k_shot)

        return prototypes

    def get_distance(self, x: torch.FloatTensor, y: torch.FloatTensor, distance_type, eps: float = 1e-10):
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
import torch
import higher
import numpy as np
import torch.nn.functional as F
from gbml.gbml import GBML
from net import mask_utils
from shot_aug import shot_aug
from utils import get_accuracy, AverageMeter


def random_rate_min(min_width=0.9, max_width=1.01, num_subnet=3):
    keep_rate = [min_width]
    keep_rate.extend(list(np.random.uniform(min_width, max_width, num_subnet - 1)))
    for k in range(len(keep_rate)):
        if keep_rate[k] > 0.999:
            keep_rate[k] = 0.999
    return keep_rate


class MAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        return None

    def outer_loop(self, batch, is_train):

        self.network.zero_grad()

        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)
        loss_log = AverageMeter()
        acc_log = AverageMeter()

        if is_train:
            self.outer_optimizer.zero_grad()
            for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
                with higher.innerloop_ctx(self.network, self.inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for step in range(self.args.n_inner):
                        train_logit = fmodel(train_input)
                        inner_loss = F.cross_entropy(train_logit, train_target)
                        diffopt.step(inner_loss)

                    test_logit = fmodel(test_input)
                    outer_loss = F.cross_entropy(test_logit, test_target)
                    loss_log.update(outer_loss.item())
                    with torch.no_grad():
                        acc_log.update(get_accuracy(test_logit, test_target).item())
                    outer_loss.backward()

                if 'mask' in self.args.net_aug:
                    width_mult_list = sorted(
                        random_rate_min(self.args.min_width, self.args.max_width, num_subnet=self.args.num_subnet),
                        reverse=True)
                    keep_mask_list = mask_utils.get_mask_list(width_mult_list, self.network, self.args.net_aug,
                                                              train_input, train_target, test_input, test_target)
                    for keep_mask in keep_mask_list:
                        [train_input_aug, train_target_aug, test_input_aug,
                         test_target_aug] = shot_aug(self.args, train_input, train_target, test_input, test_target)

                        with higher.innerloop_ctx(self.network, self.inner_optimizer, copy_initial_weights=False) as (
                                fmodel, diffopt):
                            for step in range(self.args.n_inner):
                                mask_utils.apply_weight_mask(fmodel, keep_mask)
                                handle_inner = mask_utils.apply_grad_mask(fmodel, keep_mask)
                                train_logit = fmodel(train_input_aug)
                                inner_loss = F.cross_entropy(train_logit, train_target_aug)
                                mask_utils.remove_weight_mask(fmodel)
                                diffopt.step(inner_loss)
                                mask_utils.remove_grad_mask(handle_inner)

                            mask_utils.apply_weight_mask(fmodel, keep_mask)
                            handle = mask_utils.apply_grad_mask(fmodel, keep_mask)
                            test_logit = fmodel(test_input_aug)
                            outer_loss = F.cross_entropy(test_logit, test_target_aug)
                            loss_log.update(outer_loss.item())
                            outer_loss.backward()
                            mask_utils.remove_weight_mask(fmodel)
                            mask_utils.remove_grad_mask(handle)

            self.outer_optimizer.step()
            return loss_log.avg, acc_log.avg, 0
        else:
            for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs,
                                                                            test_targets):
                with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=False) as (
                        fmodel, diffopt):
                    for step in range(self.args.n_inner):
                        train_logit = fmodel(train_input)
                        inner_loss = F.cross_entropy(train_logit, train_target)
                        diffopt.step(inner_loss)
                    test_logit = fmodel(test_input)
                    outer_loss = F.cross_entropy(test_logit, test_target)
                    loss_log.update(outer_loss.item())
                    with torch.no_grad():
                        acc_log.update(get_accuracy(test_logit, test_target).item())
            return loss_log.avg, acc_log.avg
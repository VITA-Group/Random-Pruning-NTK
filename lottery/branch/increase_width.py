import copy
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel

class Branch(base.Branch):
    @staticmethod
    def description():
        return "increase the width of the source network."

    @staticmethod
    def name():
        return 'increase_width'

    def branch_function(self, target_model_name: str = None, widen_ratio: int = 2, eps: float = 0.,
            start_at_step_zero: bool = False):
        if eps > 1/2 or eps < 0: raise RuntimeError("eps needs to be between 0 and 1/2 (inclusively).")
        if (('cifar' in target_model_name and 'resnet' in target_model_name) or 
        ('imagenet' in target_model_name and 'resnet' in target_model_name) or 
        ('cifar' in target_model_name and 'vggnfc' in target_model_name) or 
        ('cifar' in target_model_name and 'vgg' in target_model_name) or 
        ('cifar' in target_model_name and 'mobilenetv1' in target_model_name) or
        ('mnist' in target_model_name and 'lenet' in target_model_name)):
            pass
        else:
            raise NotImplementedError('Other mapping cases not implemented yet')

        # Load source model at `train_start_step`
        src_mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0it') if start_at_step_zero else self.lottery_desc.train_start_step
        # model = PrunedModel(models.registry.get(self.lottery_desc.model_hparams), src_mask)
        src_model = models.registry.load(self.level_root, start_step, self.lottery_desc.model_hparams)

        # create target model
        target_model_hparams = copy.deepcopy(self.lottery_desc.model_hparams)
        target_model_hparams.model_name = target_model_name
        target_model = models.registry.get(target_model_hparams)
        target_ones_mask = Mask.ones_like(target_model)



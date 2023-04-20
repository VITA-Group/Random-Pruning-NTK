import torch
import argparse
import models.registry
from training.train import standard_train
from training.runner import TrainingRunner
from training.desc import TrainingDesc
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
import platforms.registry

def main():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--directory", type=str, required=True, help="Provide the directory of the save model and mask.")
    parser.add_argument("--model_filename", type=str, required=True, help="The file name of the saved model.")
    TrainingRunner.add_args(parser)
    platforms.registry.get('local').add_args(parser)
    args = parser.parse_args()
    trainingDesc = TrainingDesc.create_from_args(args)
    model = models.registry.get(trainingDesc.model_hparams)
    model.load_state_dict(torch.load(args.directory + "/" + args.model_filename))
    mask = Mask.load(args.directory + "/mask.pth")
    prunedModel = PrunedModel(model, mask)
    standard_train(prunedModel, args.directory + "/lottery_ticket_performance", trainingDesc.dataset_hparams, trainingDesc.training_hparams)
    
if __name__ == '__main__':
    main()

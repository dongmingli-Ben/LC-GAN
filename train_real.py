import argparse
import torch
from utils.dataloader import get_real_dataloaders
from models.Classifier import RealClassifier
from utils.Trainer import TrainerReal

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='CelebA',
                        help='Base dir of the CelebA dataset')
    parser.add_argument('--latent-dir', type=str, default='latent',
                        help='Directory for extracted latents')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--negative-ratio', type=int, default=1,
                        help='Number of negative samples for one positive'
                        ' latent vector.')
    
    # Learning
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--early-stop', type=int, default=-1,
                        help='epochs for early stopping (negative number'
                        ' means not using early stopping)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Threshold for early stopping, used to quicken'
                        'early stop')
    
    # output and log
    parser.add_argument('--save-dir', type=str, default='checkpoint')
    parser.add_argument('--log-period', type=int, default=1000)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dl, valid_dl, test_dl = get_real_dataloaders(args)

    real_model = RealClassifier()

    trainer = TrainerReal(args, train_dl, valid_dl, test_dl, real_model)

    trainer.fit()

if __name__ == '__main__':
    main()
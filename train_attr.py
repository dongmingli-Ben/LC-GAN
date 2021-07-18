import argparse
import torch
from utils.dataloader import get_attr_dataloaders
from models.Classifier import RealClassifier
from utils.Trainer import TrainerAttr

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='CelebA',
                        help='Base dir of the CelebA dataset')
    parser.add_argument('--latent-dir', type=str, default='latent',
                        help='Directory for extracted latents')
    parser.add_argument('--annotation-name', type=str, default='list_attr_celeba_processed.txt',
                        help='File name containing the annotation')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    
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
    print('Using device', args.device)

    train_dl, valid_dl, test_dl, num_outputs = get_attr_dataloaders(args)

    attr_model = RealClassifier(num_outputs)

    trainer = TrainerAttr(args, train_dl, valid_dl, test_dl, attr_model)

    trainer.fit()

if __name__ == '__main__':
    main()
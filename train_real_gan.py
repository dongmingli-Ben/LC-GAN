import argparse
import torch
from utils.dataloader import get_real_dataloaders
from models.Classifier import RealClassifier
from models.Actors import RealActor
from utils.Trainer import TrainerRealGAN

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='CelebA',
                        help='Base dir of the CelebA dataset')
    parser.add_argument('--latent-dir', type=str, default='latent',
                        help='Directory for extracted latents')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--discriminate-ratio', type=int, default=10,
                        help='Number of discriminator (D) step for one actor'
                        ' (G) step.')
    parser.add_argument('--prior-ratio', type=float, default=0.1,
                        help='Probability of sampling from prior instead of'
                        ' transformed latent to update discriminator (D).')
    parser.add_argument('--lambda-dist', type=float, default=0.1,
                        help='Penalty of distance in the loss of G')
    parser.add_argument('--negative-ratio', type=int, default=1,
                        help='Number of negative samples from prior for one positive'
                        ' latent vector.')
    
    # Learning
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--grad-penalty', type=float, required=True, default=10,
                        help='coefficient of the gradient penalty term')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Number of steps to train')
    
    # output and log
    parser.add_argument('--save-dir', type=str, default='checkpoint')
    parser.add_argument('--log-period', type=int, default=1000)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device', args.device)

    train_dl, valid_dl, test_dl = get_real_dataloaders(args)

    real_model = RealClassifier()
    real_actor = RealActor()

    trainer = TrainerRealGAN(args, train_dl, real_model, real_actor)
    trainer.load_model()

    trainer.fit()

if __name__ == '__main__':
    main()
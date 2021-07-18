import argparse
import torch
from utils.dataloader import get_extract_dataloaders
from models.Vanilla_VAE import VAE
from utils.Trainer import TrainerExtract

def load_model(model, path, device):
    state_dict = torch.load(path, map_location=device)['model']
    model.load_state_dict(state_dict)

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='CelebA',
                        help='Base dir of the CelebA dataset')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Model
    parser.add_argument('--model-path', type=str, required=True, 
                        default='save/vae-0.01/best_model-15500.9935.pt',
                        help='Path of the trained VAE.')
    
    # output and log
    parser.add_argument('--save-dir', type=str, default='latent')
    parser.add_argument('--log-period', type=int, default=1000)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dl, valid_dl, test_dl = get_extract_dataloaders(args)

    vae_model = VAE()
    load_model(vae_model, args.model_path, args.device)

    trainer = TrainerExtract(args, train_dl, valid_dl, test_dl, vae_model)

    trainer.infer()

if __name__ == '__main__':
    main()
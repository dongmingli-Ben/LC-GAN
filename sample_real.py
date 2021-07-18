import argparse
import os
from models.Vanilla_VAE import VAE
from models.Classifier import RealClassifier
from models.Actors import RealActor
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_model(model, path, device, model_name='model'):
    state_dict = torch.load(path, map_location=device)[model_name]
    model.load_state_dict(state_dict)
    model.to(device)
    print('Model loaded from', path)

def freeze(model):
    for p in model.parameters():
        p.require_grad = False

def parse_args():
    parser = argparse.ArgumentParser()

    # Model path
    parser.add_argument('--vae-path', type=str, required=True,
                        help='Path to the trained VAE model')
    parser.add_argument('--real-path', type=str, required=True,
                        help='Path to the trained real classifier')

    # Generation setting
    parser.add_argument('--output-dir', type=str, default='samples-real',
                        help='Output directory for generated images')
    parser.add_argument('--num', type=int, default=5,
                        help='Number of images to be generated')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Threshold for generating images')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Ratio to perform gradient ascent')
    parser.add_argument('--actor', action='store_true',
                        help='Whether to use amortized actor')
    
    args = parser.parse_args()

    return args

def move_latent(z, discriminator, threshold, lr):
    z = nn.Parameter(z)
    optimizer = torch.optim.Adam([z], lr=lr)
    step = 0
    # import pdb; pdb.set_trace()
    while True:
        logit = discriminator(z, return_logit=True).squeeze()
        prob = torch.sigmoid(logit).item()
        if prob >= threshold:
            break
        target = torch.ones_like(logit)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    print('Stop optimizing for', step, 'step',
          'real probability', prob)
    return z

def main():
    args = parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device', args.device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    vae_model = VAE()
    classifier = RealClassifier()
    load_model(vae_model, args.vae_path, args.device)
    load_model(classifier, args.real_path, args.device, 'd_model')
    # load_model(classifier, args.real_path, args.device)

    freeze(vae_model)
    freeze(classifier)
    if args.actor:
        print('Using Actor ...')
        actor = RealActor()
        load_model(actor, args.real_path, args.device, 'g_model')
        freeze(actor)
    # import pdb; pdb.set_trace()
    for i in range(args.num):
        prior = torch.randn(1, 1024).to(args.device)
        origin = torch.clone(prior)
        if args.actor:
            prior = actor(prior)
        else:
            prior = move_latent(prior, classifier, args.threshold, args.lr)
        dist = torch.norm(origin-prior, dim=1).item()
        print('Distance', dist)
        with torch.no_grad():
            image = vae_model.decode(prior)[0]
            image = image.permute(1, 2, 0).cpu().numpy()
        path = os.path.join(args.output_dir, f'{i}.png')
        plt.imsave(path, image)

if __name__ == '__main__':
    main()
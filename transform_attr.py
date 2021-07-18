import argparse
import os

import numpy as np
from models.Vanilla_VAE import VAE
from models.Classifier import RealClassifier
from models.Actors import RealActor
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

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
    parser.add_argument('--attr-path', type=str, required=True,
                        help='Path to the trained attribute classifier')

    # Generation setting
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='samples-transform',
                        help='Output directory for generated images')
    parser.add_argument('--threshold-attr', type=float, default=0.8,
                        help='Attribute threshold for generating images')
    parser.add_argument('--threshold-real', type=float, default=0.8,
                        help='Real threshold for generating images')
    parser.add_argument('--num', type=int, default=5,
                        help='Number of images to generate for one input')
    parser.add_argument('--attr-index', type=int, default=0,
                        help='The index of attribute to be transformed into')
    parser.add_argument('--has-attr', type=int, default=1,
                        help='The attribute to be transformed into (0 means to reverse)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Ratio to perform gradient ascent')
    
    args = parser.parse_args()

    return args

def move_latent(z, real_disc, attr_disc, threshold_real, threshold_attr, lr, attr_index, has_attr):
    z = nn.Parameter(z)
    optimizer = torch.optim.Adam([z], lr=lr)
    step = 0
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        logit_real = real_disc(z, return_logit=True).squeeze()
        logit_attr = attr_disc(z, return_logit=True).squeeze()[attr_index]
        prob_real = torch.sigmoid(logit_real).item()
        prob_attr = torch.sigmoid(logit_attr).item()
        if not has_attr: prob_attr = 1 - prob_attr
        print('Initial real probability', prob_real,
                'attr probability', prob_attr)

    while True:
        logit_real = real_disc(z, return_logit=True).squeeze()
        logit_attr = attr_disc(z, return_logit=True).squeeze()[attr_index]
        prob_real = torch.sigmoid(logit_real).item()
        prob_attr = torch.sigmoid(logit_attr).item()
        if not has_attr: prob_attr = 1 - prob_attr
        if prob_real >= threshold_real and prob_attr >= threshold_attr:
            break
        target_real = torch.ones_like(logit_real)
        target_attr = torch.ones_like(logit_attr).fill_(has_attr)
        loss_real = F.binary_cross_entropy_with_logits(logit_real, target_real)
        loss_attr = F.binary_cross_entropy_with_logits(logit_attr, target_real)
        loss = loss_attr + 0.1*loss_real
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    print('Stop optimizing for', step, 'step',
          'real probability', prob_real,
          'attr probability', prob_attr)
    return z

def main():
    args = parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device', args.device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    vae_model = VAE()
    real_classifier = RealClassifier()
    attr_classifier = RealClassifier(7)
    load_model(vae_model, args.vae_path, args.device)
    load_model(real_classifier, args.real_path, args.device)
    load_model(attr_classifier, args.attr_path, args.device)

    freeze(vae_model)
    freeze(real_classifier)
    freeze(attr_classifier)
    # import pdb; pdb.set_trace()
    image = Image.open(args.input_path)
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        image = transform(image).to(args.device)
        plt.imsave(os.path.join(args.output_dir, 'origin.png'), image.permute(1, 2, 0).cpu().numpy())
        mu, std, recon = vae_model(image.unsqueeze(0))
        recon = recon[0]
        plt.imsave(os.path.join(args.output_dir, 'reconstruct.png'), recon.permute(1, 2, 0).cpu().numpy())
    for i in range(args.num):
        prior = vae_model.sample(mu, std)
        origin = torch.clone(prior)
        prior = move_latent(prior, real_classifier, attr_classifier, 
                            args.threshold_real, args.threshold_attr, 
                            args.lr, args.attr_index, args.has_attr)
        dist = torch.norm(origin-prior, dim=1).item()
        print('Distance', dist)
        with torch.no_grad():
            image = vae_model.decode(prior)[0]
            image = image.permute(1, 2, 0).cpu().numpy()
        path = os.path.join(args.output_dir, f'{i}.png')
        plt.imsave(path, image)

if __name__ == '__main__':
    main()
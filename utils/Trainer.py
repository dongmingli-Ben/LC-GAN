import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from random import random

def KL(mu, std):
    var = torch.pow(std, 2)
    res = torch.pow(mu, 2) + var - 1 - 2*torch.log(std)
    res = torch.sum(res, dim=-1)
    return torch.mean(res) / 2

def make_reconstruction_loss(var):

    def likelihood(x_real, x_recon):
        batch_size = x_real.size(0)
        x_real = x_real.view(batch_size, -1)
        x_recon = x_recon.view(batch_size, -1)
        length = x_real.size(-1)
        log_p = -length * log(2*pi) / 2 - length * log(var) / 2 \
            - F.mse_loss(x_recon, x_real) * length / (2*var)
        return log_p

    return likelihood

class Trainer:

    def __init__(self, args, train_dl, valid_dl, test_dl, model):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.device = args.device
        self.model = model.to(self.device)
        self.save_dir = args.save_dir
        self.max_epoch = args.max_epoch
        self.early_stop = args.early_stop
        self.log_period = args.log_period
        self.threshold = args.threshold
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.log_p = make_reconstruction_loss(args.var)
        self.KL = KL

    def fit(self):
        max_elbo = -1e9
        patient = None if self.early_stop < 0 else self.early_stop
        for epoch in range(self.max_epoch):
            print('Epoch', epoch, 'Maximum Epochs', self.max_epoch)
            self.model.train()
            train_elbo, train_ll, train_kl = 0, 0, 0
            for i, images in enumerate(self.train_dl):
                images = images.to(self.device)
                mu, std, x_recon = self.model(images)
                ll = self.log_p(images, x_recon)
                kl = self.KL(mu, std)
                elbo = ll - kl
                train_elbo += elbo.item()
                train_ll += ll.item()
                train_kl += kl.item()
                loss = -elbo
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.train_dl):6d}] ELBO {train_elbo/(i+1):.4f} '
                          f'KL {train_kl/(i+1):.4f} Likelihood (reconstruction) {train_ll/(i+1):.4f}')
            self.save_checkpoint()

            self.model.eval()
            with torch.no_grad():
                valid_elbo, valid_ll, valid_kl = 0, 0, 0
                for i, images in enumerate(self.valid_dl):
                    images = images.to(self.device)
                    mu, std, x_recon = self.model(images)
                    ll = self.log_p(images, x_recon)
                    kl = self.KL(mu, std)
                    elbo = ll - kl
                    valid_elbo += elbo.item()
                    valid_ll += ll.item()
                    valid_kl += kl.item()
                    if i % self.log_period == 0:
                        print(f'[{i:6d}/{len(self.valid_dl):6d}] ELBO {valid_elbo/(i+1):.4f} '
                              f'KL {valid_kl/(i+1):.4f} Likelihood (reconstruction) {valid_ll/(i+1):.4f}')
                print(f'Validation ELBO {valid_elbo/(len(self.valid_dl)):.4f} '
                      f'KL {valid_kl/(len(self.valid_dl)):.4f} Likelihood (reconstruction)'
                      f' {valid_ll/(len(self.valid_dl)):.4f}')
                elbo = valid_elbo / len(self.valid_dl)
                if not patient is None:
                    patient -= 1
                    if elbo > max_elbo + self.threshold:
                        max_elbo = elbo
                        patient = self.early_stop
                        self.save_checkpoint(elbo)
                    if patient < 0:
                        print('Early stopping ...')
                        break

        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            test_elbo, test_ll, test_kl = 0, 0, 0
            for i, images in enumerate(self.test_dl):
                images = images.to(self.device)
                mu, std, x_recon = self.model(images)
                ll = self.log_p(images, x_recon)
                kl = self.KL(mu, std)
                elbo = ll - kl
                test_elbo += elbo.item()
                test_ll += ll.item()
                test_kl += kl.item()
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.test_dl):6d}] ELBO {test_elbo/(i+1):.4f} '
                          f'KL {test_kl/(i+1):.4f} Likelihood (reconstruction) {test_ll/(i+1):.4f}')
            print(f'Test ELBO {test_elbo/(len(self.test_dl)):.4f} '
                  f'KL {test_kl/(len(self.test_dl)):.4f} Likelihood (reconstruction)'
                  f' {test_ll/(len(self.test_dl)):.4f}')

    def evaluate(self, dataloader, load_model=True):
        if load_model:
            self.load_best_model()
        self.model.eval()
        with torch.no_grad():
            test_elbo, test_ll, test_kl = 0, 0, 0
            for i, images in enumerate(dataloader):
                images = images.to(self.device)
                mu, std, x_recon = self.model(images)
                ll = self.log_p(images, x_recon)
                kl = self.KL(mu, std)
                elbo = ll - kl
                test_elbo += elbo.item()
                test_ll += ll.item()
                test_kl += kl.item()
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(dataloader):6d}] ELBO {test_elbo/(i+1):.4f} '
                          f'KL {test_kl/(i+1):.4f} Likelihood (reconstruction) {test_ll/(i+1):.4f}')
            print(f'Test ELBO {test_elbo/(len(dataloader)):.4f} '
                  f'KL {test_kl/(len(dataloader)):.4f} Likelihood (reconstruction)'
                  f' {test_ll/(len(dataloader)):.4f}')


    def save_checkpoint(self, elbo=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('Saving model ...')
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if elbo is None:
            path = os.path.join(self.save_dir, 'current_checkpoint.pt')
            torch.save(state_dict, path)
        else:
            for root, directory, files in os.walk(self.save_dir):
                for file in files:
                    if 'best_model' in file:
                        os.remove(os.path.join(root, file))
            path = os.path.join(self.save_dir, f'best_model-{elbo}.pt')
            torch.save(state_dict, path)
        print('Model saved to', path)

    def load_best_model(self):
        for root, directory, files in os.walk(self.save_dir):
            for file in files:
                if 'best_model' in file:
                    path = os.path.join(root, file)
                    break
        print(f'Loading model from {path} ...')
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        print('Model loaded.')


class TrainerExtract:

    def __init__(self, args, train_dl, valid_dl, test_dl, model):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.device = args.device
        self.model = model.to(self.device)
        self.save_dir = args.save_dir
        self.log_period = args.log_period
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def infer(self):
        self.extract(self.train_dl)
        self.extract(self.valid_dl)
        self.extract(self.test_dl)

    def extract(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                images = data['image']
                images = images.to(self.device)
                mu, std = self.model.encode(images)
                self.save_to_files(mu, std, data['path'])
                if i % self.log_period == 0:
                    print('Processing', i+1, 'total', len(dataloader))

    def save_to_files(self, mu, std, paths):
        for m, s, path in zip(mu, std, paths):
            path_ = os.path.join(self.save_dir, path.split('/')[-1].replace('jpg', 'pt'))
            torch.save(torch.stack([m, s]), path_)


class TrainerReal(Trainer):

    def __init__(self, args, train_dl, valid_dl, test_dl, model):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.device = args.device
        self.model = model.to(self.device)
        self.save_dir = args.save_dir
        self.max_epoch = args.max_epoch
        self.early_stop = args.early_stop
        self.log_period = args.log_period
        self.threshold = args.threshold
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.BCELoss()

    def fit(self):
        max_acc = 0
        patient = None if self.early_stop < 0 else self.early_stop
        for epoch in range(self.max_epoch):
            print('Epoch', epoch, 'Maximum Epochs', self.max_epoch)
            self.model.train()
            train_loss = 0
            for i, data in enumerate(self.train_dl):
                samples = data['samples'].to(self.device)
                labels = data['labels'].to(self.device)
                probs = self.model(samples)
                loss = self.loss(probs.squeeze(), labels)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.train_dl):6d}] Loss {train_loss/(i+1):.4f} ')
            self.save_checkpoint()

            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                correct, total = 0, 0
                for i, data in enumerate(self.valid_dl):
                    samples = data['samples'].to(self.device)
                    labels = data['labels'].to(self.device)
                    probs = self.model(samples)
                    loss = self.loss(probs.squeeze(), labels)
                    valid_loss += loss.item()
                    pred = (probs.squeeze() > 0.5)
                    correct += (labels == pred).sum().item()
                    total += labels.size(0)
                    if i % self.log_period == 0:
                        print(f'[{i:6d}/{len(self.valid_dl):6d}] Loss {valid_loss/(i+1):.4f} '
                              f'Accuracy {correct/total:.4f}')
                print(f'Validation Loss {valid_loss/(len(self.valid_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')
                acc = correct / total
                if not patient is None:
                    patient -= 1
                    if acc > max_acc + self.threshold:
                        max_acc = acc
                        patient = self.early_stop
                        self.save_checkpoint(acc)
                    if patient < 0:
                        print('Early stopping ...')
                        break

        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            test_loss = 0
            correct, total = 0, 0
            for i, data in enumerate(self.test_dl):
                samples = data['samples'].to(self.device)
                labels = data['labels'].to(self.device)
                probs = self.model(samples)
                loss = self.loss(probs.squeeze(), labels)
                test_loss += loss.item()
                pred = (probs.squeeze() > 0.5)
                correct += (labels == pred).sum().item()
                total += labels.size(0)
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.test_dl):6d}] Loss {test_loss/(i+1):.4f} '
                          f'Accuracy {correct/total:.4f}')
            print(f'Validation Loss {test_loss/(len(self.test_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')

    def evaluate(self, dataloader, load_model=True):
        if load_model:
            self.load_best_model()
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            correct, total = 0, 0
            for i, data in enumerate(self.test_dl):
                samples = data['samples'].to(self.device)
                labels = data['labels'].to(self.device)
                probs = self.model(samples)
                loss = self.loss(probs.squeeze(), labels)
                test_loss += loss.item()
                pred = (probs.squeeze() > 0.5)
                correct += (labels == pred).sum().item()
                total += labels.size(0)
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.test_dl):6d}] Loss {test_loss/(i+1):.4f} '
                          f'Accuracy {correct/total:.4f}')
            print(f'Validation Loss {test_loss/(len(self.test_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')

def make_d1_loss():

    def loss(logits: torch.Tensor):
        target = torch.ones_like(logits)
        res = F.binary_cross_entropy_with_logits(logits, target)
        return res

    return loss

def make_d0_loss():

    def loss(logits: torch.Tensor):
        target = torch.zeros_like(logits)
        res = F.binary_cross_entropy_with_logits(logits, target)
        return res

    return loss

def make_g_loss(dataloader, d1_loss, lambda_dist, discriminator):
    total_std, num = 0, 0
    for data in dataloader:
        std = data['std']
        total_std += torch.sum(std, 0)
        num += std.size(0)
    avg_std = (total_std / num).mean()
    inv_var = torch.div(1, avg_std**2).item()
    print('Avg 1/var', inv_var)

    def loss(transformed, origin):
        res = d1_loss(discriminator(transformed, return_logit=True))
        dist = torch.norm(transformed-origin, dim=-1)
        l_dist = inv_var * torch.log1p(dist**2).mean()
        res += lambda_dist*l_dist
        return res, dist.mean().item()

    return loss

def compute_grad_penalty(discriminator, real_samples, fake_samples, device):
    """Compute the gradient penalty for regularization. Intuitively, the
    gradient penalty help stablize the magnitude of the gradients that the
    discriminator provides to the generator, and thus help stablize the training
    of the generator."""
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)
    # Get gradients w.r.t. the interpolations
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class TrainerRealGAN:

    def __init__(self, args, train_dl, d_model, g_model):
        self.train_dl = train_dl
        self.device = args.device
        self.d_model = d_model.to(self.device)
        self.g_model = g_model.to(self.device)
        self.save_dir = args.save_dir
        self.max_steps = args.max_steps
        self.log_period = args.log_period
        self.d_steps = args.discriminate_ratio
        self.prior_prob = args.prior_ratio
        self.grad_penalty = args.grad_penalty
        self.lr = args.lr
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.lr)
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.lr)
        self.d1_loss = make_d1_loss()
        self.d0_loss = make_d0_loss()
        self.g_loss = make_g_loss(self.train_dl, self.d1_loss, args.lambda_dist, self.d_model)

    def fit(self):
        self.d_model.train()
        self.g_model.train()
        steps = 0
        g_loss_sum, d_loss_sum, grad_penalty_sum, dist_sum = 0, 0, 0, 0
        # import pdb; pdb.set_trace()
        while steps < self.max_steps:
            for i, data in enumerate(self.train_dl):
                steps += 1
                # Discriminator
                self.d_optimizer.zero_grad()
                real_z = data['z'].to(self.device)
                fake_z = torch.randn_like(real_z)
                if random() < self.prior_prob:
                    transformed_z = torch.randn_like(real_z)
                else:
                    transformed_z = self.g_model(fake_z).detach()
                d_loss = self.d1_loss(self.d_model(real_z, return_logit=True)) \
                    + self.d0_loss(self.d_model(fake_z, return_logit=True)) \
                    + self.d0_loss(self.d_model(transformed_z, return_logit=True))
                d_loss_sum += d_loss.item()
                d_loss.backward()

                # gradient penalty
                gradient_penalty = self.grad_penalty * compute_grad_penalty(
                    self.d_model, real_z, transformed_z, self.device
                )
                grad_penalty_sum += gradient_penalty.item()
                gradient_penalty.backward()

                torch.nn.utils.clip_grad_value_(self.d_model.parameters(), 0.1)
                self.d_optimizer.step()

                # Generator
                if steps%self.d_steps == 0:
                    self.g_optimizer.zero_grad()
                    transformed_z = self.g_model(fake_z)
                    g_loss, dist = self.g_loss(transformed_z, fake_z)
                    g_loss_sum += g_loss.item()
                    dist_sum += dist
                    g_loss.backward()
                    self.g_optimizer.step()

                if steps % self.log_period == 0:
                    print(f'[{steps:6d}/{self.max_steps:6d}] Avg G Loss {g_loss_sum/self.log_period:.4f}'
                          f' Avg D Loss {d_loss_sum/self.log_period:.4f} Avg gradient penalty'
                          f' {grad_penalty_sum/self.log_period:.4f} Avg distance {dist_sum/self.log_period:.4f}')
                    g_loss_sum, d_loss_sum, grad_penalty_sum, dist_sum = 0, 0, 0, 0

                if steps >= self.max_steps:
                    break

        self.save_checkpoint()

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('Saving model ...')
        state_dict = {
            'd_model': self.d_model.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_model': self.g_model.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
        }
        path = os.path.join(self.save_dir, 'current_checkpoint.pt')
        torch.save(state_dict, path)
        print('Model saved to', path)

    def load_model(self):
        for root, directory, files in os.walk(self.save_dir):
            for file in files:
                if 'current_checkpoint' in file:
                    path = os.path.join(root, file)
                    break
        print(f'Loading model from {path} ...')
        state_dict = torch.load(path)
        self.d_model.load_state_dict(state_dict['d_model'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer'])
        self.g_model.load_state_dict(state_dict['g_model'])
        self.g_optimizer.load_state_dict(state_dict['g_optimizer'])
        print('Model loaded.')


class TrainerAttr(Trainer):

    def __init__(self, args, train_dl, valid_dl, test_dl, model):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.device = args.device
        self.model = model.to(self.device)
        self.save_dir = args.save_dir
        self.max_epoch = args.max_epoch
        self.early_stop = args.early_stop
        self.log_period = args.log_period
        self.threshold = args.threshold
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.BCELoss()

    def fit(self):
        max_acc = 0
        patient = None if self.early_stop < 0 else self.early_stop
        for epoch in range(self.max_epoch):
            print('Epoch', epoch, 'Maximum Epochs', self.max_epoch)
            self.model.train()
            train_loss = 0
            for i, data in enumerate(self.train_dl):
                samples = data['z'].to(self.device)
                labels = data['attr_labels'].to(self.device).float()
                probs = self.model(samples)
                loss = self.loss(probs, labels)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.train_dl):6d}] Loss {train_loss/(i+1):.4f} ')
            self.save_checkpoint()

            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                correct, total = 0, 0
                for i, data in enumerate(self.valid_dl):
                    samples = data['z'].to(self.device)
                    labels = data['attr_labels'].to(self.device).float()
                    probs = self.model(samples)
                    loss = self.loss(probs, labels)
                    valid_loss += loss.item()
                    pred = (probs > 0.5)
                    correct += (labels == pred).sum().item()
                    total += torch.numel(labels)
                    if i % self.log_period == 0:
                        print(f'[{i:6d}/{len(self.valid_dl):6d}] Loss {valid_loss/(i+1):.4f} '
                              f'Accuracy {correct/total:.4f}')
                print(f'Validation Loss {valid_loss/(len(self.valid_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')
                acc = correct / total
                if not patient is None:
                    patient -= 1
                    if acc > max_acc + self.threshold:
                        max_acc = acc
                        patient = self.early_stop
                        self.save_checkpoint(acc)
                    if patient < 0:
                        print('Early stopping ...')
                        break

        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            test_loss = 0
            correct, total = 0, 0
            for i, data in enumerate(self.test_dl):
                samples = data['z'].to(self.device)
                labels = data['attr_labels'].to(self.device).float()
                probs = self.model(samples)
                loss = self.loss(probs, labels)
                test_loss += loss.item()
                pred = (probs > 0.5)
                correct += (labels == pred).sum().item()
                total += torch.numel(labels)
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.test_dl):6d}] Loss {test_loss/(i+1):.4f} '
                          f'Accuracy {correct/total:.4f}')
            print(f'Validation Loss {test_loss/(len(self.test_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')

    def evaluate(self, dataloader, load_model=True):
        if load_model:
            self.load_best_model()
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            correct, total = 0, 0
            for i, data in enumerate(self.test_dl):
                samples = data['z'].to(self.device)
                labels = data['attr_labels'].to(self.device).float()
                probs = self.model(samples)
                loss = self.loss(probs, labels)
                test_loss += loss.item()
                pred = (probs > 0.5)
                correct += (labels == pred).sum().item()
                total += torch.numel(labels)
                if i % self.log_period == 0:
                    print(f'[{i:6d}/{len(self.test_dl):6d}] Loss {test_loss/(i+1):.4f} '
                          f'Accuracy {correct/total:.4f}')
            print(f'Validation Loss {test_loss/(len(self.test_dl)):.4f} '
                      f'Accuracy {correct/total:.4f}')
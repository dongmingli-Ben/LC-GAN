import torch
import matplotlib.pyplot as plt
from models.Vanilla_VAE import VAE
from PIL import Image
from torchvision import transforms

def load_model(model, path, device):
    state_dict = torch.load(path, map_location=device)['model']
    model.load_state_dict(state_dict)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
model = VAE()
# load_model(model, 'save/vae-0.0001/best_model-9937.3093.pt', device)
load_model(model, 'save/vae-0.001/best_model-25711.7383.pt', device)
# load_model(model, 'save/vae/best_model--20813.25173611111.pt', device)
# load_model(model, 'save/vae-0.1/best_model-16602.910856827446.pt', device)

N = 5
for i in range(N):
    z = torch.randn((1, 1024)).to(device)
    with torch.no_grad():
        image = model.decode(z).cpu()[0]
        image = image.permute(1, 2, 0).numpy()
    plt.imsave(f'samples/sample-{i}-0.001.png', image)

transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize(64),
    transforms.ToTensor(),
])

image = Image.open('CelebA/Img/120537.jpg')
image = transform(image)
with torch.no_grad():
    _, _, recon = model(image.unsqueeze(0).to(device))
image = image.permute(1, 2, 0)
plt.imsave('samples/image.png', image.numpy())
recon = recon[0].permute(1, 2, 0).cpu()
plt.imsave('samples/recon-0.001.png', recon.numpy())

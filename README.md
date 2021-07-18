# LC-GAN

## Requirements

* pytorch
* PIL

## How to Run

For datasets, you can download the CelebA dataset [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Example scripts are provided for easy use.

|File|Usage|
|-|-|
|```run.sh```|Train a VAE|
|```extract.sh```|Save the posterior of the images in the datset using a trained VAE in ```run.sh```|
|```real.sh```|Train a realism classifier without using GAN|
|```real_gan.sh```|Train a realism classifier using GAN|
|```attr.sh```|Train a attribute classifier without using GAN|
|```generate_real.sh```|Generate realistic images using VAE, realism classifier or realism actor|
|```transform.sh```|Perform identity-preserved transformation using VAE, realism classifier and attribute classifier|

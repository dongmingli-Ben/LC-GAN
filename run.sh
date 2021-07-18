var=0.001

python train_vae.py \
--data-dir CelebA \
--batch-size 16 \
--lr 3e-4 \
--var ${var} \
--early-stop 5 \
--save-dir save/vae-${var}
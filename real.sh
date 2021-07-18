ns=1

python train_real.py \
--data-dir CelebA \
--latent-dir latent \
--batch-size 16 \
--lr 3e-4 \
--negative-ratio ${ns} \
--early-stop 5 \
--save-dir save/real-${ns}
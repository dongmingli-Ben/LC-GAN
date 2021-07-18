python train_real_gan.py \
--data-dir CelebA \
--latent-dir latent \
--batch-size 16 \
--num-workers 0 \
--discriminate-ratio 10 \
--prior-ratio 0.1 \
--lambda-dist 0.1 \
--lr 3e-4 \
--grad-penalty 10 \
--max-steps 200000 \
--save-dir save/real-gan \
--log-period 500
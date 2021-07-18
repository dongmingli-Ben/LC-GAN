ns=1

python train_attr.py \
--data-dir CelebA \
--latent-dir latent \
--annotation-name list_attr_celeba_processed.txt \
--batch-size 16 \
--lr 3e-4 \
--early-stop 5 \
--save-dir save/attr
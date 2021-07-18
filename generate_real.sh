python sample_real.py \
--vae-path save/vae-0.01/best_model-15500.9935.pt \
--real-path save/real-gan/current_checkpoint-200000.pt \
--lr 1e-2 \
--output-dir samples-real \
--num 5 \
--threshold 0.9 \
--actor
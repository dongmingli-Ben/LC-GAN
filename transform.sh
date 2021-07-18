threshold=0.97

python transform_attr.py \
--vae-path save/vae-0.01/best_model-15500.9935.pt \
--real-path save/real-1/best_model-0.9999748326370362.pt \
--attr-path save/attr/best_model-0.9206077558621979.pt \
--input-path image.jpg \
--output-dir samples-transform \
--lr 1e-2 \
--threshold-attr ${threshold} \
--threshold-real ${threshold} \
--attr-index 3 \
--has-attr 0 \
--num 5 \
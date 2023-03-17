python AIR.py --dataset Beibei --data_path Data/ \
--regs [1e-4] \
--embed_size 64 \
--lr 0.001 \
--batch_size 2048 \
--epoch 1000 \
--neg_num 4 \
--model_type air_normal

python AIR_rel_u.py --dataset Beibei --data_path Data/ \
--regs [1e-4] \
--embed_size 64 \
--lr 0.001 \
--batch_size 2048 \
--epoch 1000 \
--neg_num 4 \
--model_type air_rel_u

python AIR_rel_ui.py --dataset Beibei --data_path Data/ \
--regs [1e-4] \
--embed_size 64 \
--lr 0.001 \
--batch_size 2048 \
--epoch 1000 \
--neg_num 4 \
--model_type air_rel_ui
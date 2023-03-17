# if u dot i 
python eval.py \
--data_path Data/ \
--dataset Beibei \
--Ks [10,50,100] \
--user_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/U_1000000 \
--item_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/I_1000000 \

# if (u+r) dot (i+r)
python eval.py \
--data_path Data/ \
--dataset Beibei \
--Ks [10,50,100] \
--user_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/U_1000000 \
--item_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/I_1000000 \
--user_rel_flag 1 \
--user_rel_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/UR_1000000 \
--item_rel_flag 1 \
--item_rel_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/IR_1000000 \

# if (u+r) dot i : 
python eval.py \
--data_path Data/ \
--dataset Beibei \
--Ks [10,50,100] \
--user_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/U_1000000 \
--item_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/I_1000000 \
--user_rel_flag 1 \
--user_rel_emb /tmp2/weile/AIR_python/AIR_handcraft/embedding/ui/UR_1000000 

# Note
# user_rel_flag = 0 , 沒有user的relation emb (只有 u dot i )
# user_rel_flag = 1 , 算 (u dot i) 和 (u+r) dot i

# item_rel_flag = 0 , 沒有item的relation emb 
# item_rel_flag = 1 , 算 (u dot i) 和 (u+r) dot (i+r)

# embedding path saved in cfda4

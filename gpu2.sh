# python main_nfm.py --data_name movielens --gpu 2
# python main_nfm.py --data_name lastfm --social_net --gpu 2
# python main_nfm.py --data_name delicious --social_net --gpu 2

# python main_nfm.py --data_name movielens --gpu 2 --model_type fm
# python main_nfm.py --data_name lastfm --social_net --gpu 2 --model_type fm
# python main_nfm.py --data_name delicious --social_net --gpu 2 --model_type fm

# python main_ecfkg.py --data_name movielens --gpu 2

# python main_ecfkg.py --data_name delicious --social_net --gpu 2 --lr 0.005
# python main_kgat.py --data_name delicious --social_net --gpu 2 --lr 0.005
# python main_graphrec.py --dataset delicious --gpu 2

# python main_tkgat.py --data_name delicious --gpu 2 --social_net

# python main_tkgat.py --data_name movielens --gpu 2 --conv_dim_list [100,50]
# python main_tkgat.py --data_name movielens --gpu 2 --conv_dim_list [100]

# python main_tkgat.py --data_name lastfm --gpu 2 --social_net --conv_dim_list [100,50]
# python main_tkgat.py --data_name lastfm --gpu 2 --social_net --conv_dim_list [100]

# python main_kgat.py --data_name movielens --gpu 2 --test --lr 0.0005
# python main_kgat.py --data_name lastfm --social_net --gpu 2 --test

# python main_ecfkg.py --data_name movielens --gpu 2 --test 
# python main_ecfkg.py --data_name lastfm --gpu 2 --test  --social_net
# python main_nfm.py --data_name movielens --gpu 2 --test
# python main_nfm.py --data_name lastfm --gpu 2 --test  --social_net
# python main_nfm.py --data_name movielens --gpu 2 --test --model_type fm
# python main_nfm.py --data_name lastfm --gpu 2 --test --model_type fm  --social_net

# python main_tkgat.py --data_name delicious --gpu 2 --social_net --lr 0.01

# python main_tkgat.py --data_name delicious --gpu 2 --social_net --use_pretrain 0 --lr 0.01


# python main_tkgat.py --data_name delicious_small --gpu 2 --social_net --relation_dim 25 --attention_dim 25 --lr 0.01
# python main_tkgat.py --data_name delicious_small --gpu 2 --social_net --relation_dim 10 --attention_dim 10 --lr 0.01

# python main_tkgat.py --data_name delicious_small --gpu 2 --social_net --relation_dim 25 --attention_dim 25 --lr 0.01 --test
# python main_tkgat.py --data_name delicious_small --gpu 2 --social_net --relation_dim 10 --attention_dim 10 --lr 0.01 --test

# python main_nfm.py --data_name delicious_small --gpu 2 --model_type nfm  --social_net
# python main_nfm.py --data_name delicious_small --gpu 2 --model_type fm  --social_net

# python main_kgat.py --data_name delicious_small --gpu 2 --social_net --lr 0.002

python main_tkgat.py --data_name delicious_small --gpu 2 --social_net --use_pretrain 0 --lr 0.005
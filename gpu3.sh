# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net
# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net --use_pretrain 0
# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net --relation_dim 25 --attention_dim 25
# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net --relation_dim 10 --attention_dim 10
# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net --conv_dim_list [100,50]
# python main_tkgat.py --data_name lastfm --gpu 3 --test --social_net --conv_dim_list [100]

# python main_graphrec.py --dataset lastfm --gpu 3
# python main_graphrec.py --dataset lastfm --gpu 3 --test

# python main_ecfkg.py --data_name delicious --gpu 3 --social_net --lr 0.05

# python main_tkgat.py --data_name delicious --gpu 3 --social_net --conv_dim_list [100,50] --lr 0.01
# python main_tkgat.py --data_name delicious --gpu 3 --social_net --conv_dim_list [100] --lr 0.01
# python main_tkgat.py --data_name delicious --gpu 3 --social_net --relation_dim 25 --attention_dim 25 --lr 0.01
# python main_tkgat.py --data_name delicious --gpu 3 --social_net --relation_dim 10 --attention_dim 10 --lr 0.01

# python main_tkgat.py --data_name delicious_small --gpu 3 --social_net --conv_dim_list [100,50] --lr 0.01
# python main_tkgat.py --data_name delicious_small --gpu 3 --social_net --conv_dim_list [100] --lr 0.01

# python main_tkgat.py --data_name delicious_small --gpu 3 --social_net --conv_dim_list [100,50] --lr 0.01 --test
# python main_tkgat.py --data_name delicious_small --gpu 3 --social_net --conv_dim_list [100] --lr 0.01 --test

# python main_nfm.py --data_name delicious_small --gpu 3 --model_type nfm  --social_net
# python main_kgat.py --data_name delicious_small --gpu 3 --social_net --lr 0.01

# python main_nfm.py --data_name delicious_small --gpu 3 --model_type fm  --social_net
# python main_ecfkg.py --data_name delicious_small --gpu 3 --social_net --lr 0.05

# python main_kgat.py --data_name delicious_small --gpu 3 --social_net --lr 0.005

python main_tkgat.py --data_name delicious_small --gpu 3 --social_net --use_pretrain 0 --lr 0.002
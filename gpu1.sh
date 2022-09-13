# python main_tkgat.py --data_name movielens --gpu 1
# python main_tkgat.py --data_name movielens --gpu 1 --test --use_pretrain 0
# python main_tkgat.py --data_name movielens --gpu 1 --test --relation_dim 25 --attention_dim 25
# python main_tkgat.py --data_name movielens --gpu 1 --test --relation_dim 10 --attention_dim 10
# python main_tkgat.py --data_name movielens --gpu 1 --test --conv_dim_list [100,50]
# python main_tkgat.py --data_name movielens --gpu 1 --test --conv_dim_list [100]

# python main_kgat.py --data_name delicious --gpu 1 --social_net --lr 0.01
# python main_tkgat.py --data_name delicious --gpu 1 --social_net --relation_dim 25 --attention_dim 25 --lr 0.01

# python main_tkgat.py --data_name delicious_small --gpu 1 --social_net --lr 0.01

# python main_tkgat.py --data_name delicious_small --gpu 1 --social_net --lr 0.01 --test
# python main_tkgat.py --data_name delicious_small --gpu 1 --social_net --use_pretrain 0 --lr 0.01 --test

# python main_kgat.py --data_name delicious_small --gpu 1 --social_net --lr 0.02

python main_tkgat.py --data_name delicious_small --gpu 1 --social_net --use_pretrain 0 --lr 0.001
# python main_nfm.py --data_name delicious --gpu 0 --model_type nfm  --social_net
# python main_nfm.py --data_name delicious --gpu 0 --model_type fm  --social_net

# python main_tkgat.py --data_name delicious --gpu 0 --social_net --lr 0.01 --test
# python main_tkgat.py --data_name delicious --gpu 0 --social_net --conv_dim_list [100,50] --lr 0.01 --test

# python main_tkgat.py --data_name delicious_small --gpu 0 --social_net --use_pretrain 0 --lr 0.01

# python main_nfm.py --data_name delicious_small --gpu 0 --model_type fm  --social_net

# python main_ecfkg.py --data_name delicious_small --gpu 0 --social_net --lr 0.05 --test 
# python main_nfm.py --data_name delicious_small --gpu 0 --model_type nfm  --social_net --test
# python main_nfm.py --data_name delicious_small --gpu 0 --model_type fm  --social_net --test

# python main_tkgat.py --data_name delicious_small --gpu 0 --social_net --use_pretrain 0 --lr 0.01 --test
python main_tkgat.py --data_name delicious_small --gpu 0 --social_net --use_pretrain 0 --lr 0.005 --test
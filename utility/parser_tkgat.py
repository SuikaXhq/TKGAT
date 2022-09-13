import argparse
import os

from utility.helper import get_best_model

# delicious: python main_tkgat.py --data_name delicious --social_net --gpu 2 --relation_dim 30  --attention_dim 30 --conv_dim_list [50,20,10] --cf_batch_size 1024 --kg_batch_size 2048 --test_batch_size 2048
def parse_tkgat_args(args=None):
    parser = argparse.ArgumentParser(description="Run TKGAT. (With KGAT loader)")

    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for using multi GPUs.')

    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='movielens',
                        help='Choose a dataset from {movielens, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with GloVe, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_epoch', type=int, default=0,
                        help='Epoch index of stored model.')
    parser.add_argument('--active_sampling', action='store_true',
                        help='Use active sampling strategy.')
    parser.add_argument('--social_net', action='store_true',
                        help='Whether it is a social network enhanced RS dataset.')
    parser.add_argument('--memory_efficiency', action='store_true',
                        help='Save memory usage.')
    parser.add_argument('--optimizer', nargs='?', default='Adam',
                        help='Optimizer, choose from {SGD, Adam}.')

    parser.add_argument('--cf_batch_size', type=int, default=2048,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=4096,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--entity_dim', type=int, default=100,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=50,
                        help='Relation Embedding size.')
    parser.add_argument('--attention_dim', type=int, default=50,
                        help='Attention parameter dimension.')
    parser.add_argument('--n_attention_heads', type=int, default=5,
                        help='Number of attention heads.')

    parser.add_argument('--aggregation_type', nargs='?', default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[100, 50, 25]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=50,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')
    parser.add_argument('--gpu', type=int, default=2,
                        help='index of GPU to use.')
    parser.add_argument('--K', type=int, default=20,
                        help='Calculate metric@K when evaluating.')

    parser.add_argument('--test', action='store_true',
                        help='Only test the pretrained model.')

    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    args.dump_dir = 'trained_model/TKGAT/{}'.format(args.data_name)
    save_dir = 'trained_model/TKGAT/{}/entitydim{}_relationdim{}_attdim{}_{}_{}_lr{}_pretrain{}_atthead{}/'.format(
        args.data_name, args.entity_dim, args.relation_dim, args.attention_dim, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, 1 if args.use_pretrain else 0, args.n_attention_heads)
    args.save_dir = save_dir

    result_dir = 'results/TKGAT/{}/entitydim{}_relationdim{}_attdim{}_{}_pretrain{}/'.format(
        args.data_name, args.entity_dim, args.relation_dim, args.attention_dim,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), 1 if args.use_pretrain else 0)
    args.result_dir = result_dir

    pretrain_model_path = os.path.join(save_dir, 'model_epoch{}.pth'.format(args.pretrain_model_epoch))
    # best_model_path = os.path.join(save_dir, get_best_model(args.save_dir))
    args.pretrain_model_path = pretrain_model_path

    return args



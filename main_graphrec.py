#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 30 Sep, 2019

@author: wangshuo
"""

import os
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from model.utils_graphrec import collate_fn
from model.graphrec import GraphRec
from utility.loader_graphrec import GRDataset
from utility.metrics import calc_metrics_at_k

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lastfm', help='dataset name, [lastfm/delicious]')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=100, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=1000, help='the number of epochs to train for')
parser.add_argument('--early_stop_epoch', type=int, default=50, help='the number of epochs to early stop')
parser.add_argument('--gpu', type=int, default=3, help='GPU index to use')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()
args.dataset_path = f'trained_model/GraphRec/{args.dataset}/'
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print('Loading data...')
    with open(args.dataset_path + 'dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(args.dataset_path + 'list.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)
    
    train_data = GRDataset(train_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    valid_data = GRDataset(valid_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    model = GraphRec(user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)

    if args.test:
        K = np.arange(1,21)
        if os.path.exists(os.path.join(args.dataset_path, 'preds.npy')):
            preds = np.load(os.path.join(args.dataset_path, 'preds.npy')).squeeze(-1)
            precision, recall, ndcg = validate_by_preds(preds, train_data, test_data, user_count, item_count, K)
        else:
            print('Load checkpoint and testing...')
            ckpt = torch.load(f'trained_model/GraphRec/{args.dataset}/best_checkpoint.pth.tar')
            print(f'User count: {user_count}')
            print(f'Item count: {item_count}')
            model.load_state_dict(ckpt['state_dict'])
            
            precision, recall, ndcg = validate(test_loader, model, 'test', train_data, test_data, user_count, item_count, K)
        # print('Test Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))
        if not os.path.exists(f'results/GraphRec/{args.dataset}/'):
            os.makedirs(f'results/GraphRec/{args.dataset}/')
        with open(f'results/GraphRec/{args.dataset}/test_result.tsv', mode='w') as f:
            f.write('K\tprecision@K\trecall@K\tndcg@K\n')
            for k in K:
                f.write('{}\t{}\t{}\t{}\n'.format(k, precision[k-1], recall[k-1], ndcg[k-1]))
        return

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    early_stop_epoch = args.early_stop_epoch

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)

        mae, rmse = validate(valid_loader, model, mode='valid')

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, f'trained_model/GraphRec/{args.dataset}/latest_checkpoint.pth.tar')

        if epoch == 0:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            torch.save(ckpt_dict, f'trained_model/GraphRec/{args.dataset}/best_checkpoint.pth.tar')
        else:
            early_stop_epoch -= 1

        print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}'.format(epoch, mae, rmse, best_mae))

        if early_stop_epoch == 0:
            print('Early stopped.')
            break

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (uids, iids, labels, u_items, u_users, u_users_items, i_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        
        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, u_users, u_users_items, i_users)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(uids) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model, mode='valid', train_data=None, valid_data=None, user_count=None, item_count=None, K=20):
    if mode == 'valid':
        model.eval()
        errors = []
        with torch.no_grad():
            for uids, iids, labels, u_items, u_users, u_users_items, i_users in tqdm(valid_loader):
                uids = uids.to(device)
                iids = iids.to(device)
                labels = labels.to(device)
                u_items = u_items.to(device)
                u_users = u_users.to(device)
                u_users_items = u_users_items.to(device)
                i_users = i_users.to(device)
                preds = model(uids, iids, u_items, u_users, u_users_items, i_users)
                error = torch.abs(preds.squeeze(1) - labels)
                errors.extend(error.data.cpu().numpy().tolist())
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.power(errors, 2)))
        return mae, rmse
    elif mode == 'test':
        model.eval()
        user_ids = list(valid_data.user_dict.keys())
        # user_ids_batches = [user_ids[i: i + args.batch_size] for i in range(0, len(user_ids), args.batch_size)]
        # user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
        # user_ids_batches = [d.to(device) for d in user_ids_batches]

        item_ids = torch.arange(item_count, dtype=torch.long)
        # item_ids = item_ids.to(device)
        item_id_batch_size = 512
        item_ids_batches = [item_ids[i: i + item_id_batch_size] for i in range(0, item_count, item_id_batch_size)]
        item_ids_batches = [torch.LongTensor(d).to(device) for d in item_ids_batches]
        preds = []
        i_users = [valid_data.i_users_list[item_id] for item_id in item_ids]
        max_len = max([len(i_user) for i_user in i_users])
        i_users = torch.tensor([np.pad(i_user, [[0, max_len - len(i_user)], [0, 0]]) for i_user in i_users])

        precision_k = []
        recall_k = []
        ndcg_k = []

        with torch.no_grad():
            for user_id in tqdm(user_ids):
                pred_user = []
                # test_batch = np.asarray([[user_id, item_id, valid_data.u_items_list[user_id], valid_data.u_users_list[user_id], valid_data.u_users_items_list[user_id], valid_data.i_users_list[item_id]] for item_id in item_ids])
                uids = (user_id * torch.ones(item_id_batch_size, dtype=torch.long)).to(device)
                u_items = torch.tensor(valid_data.u_items_list[user_id]).unsqueeze(0)[torch.zeros(item_id_batch_size, dtype=torch.long), ...].to(device)
                u_users = torch.tensor(valid_data.u_users_list[user_id]).unsqueeze(0)[torch.zeros(item_id_batch_size, dtype=torch.long), ...].to(device)
                u_users_items_list = valid_data.u_users_items_list[user_id]
                max_len = max([len(u_users_items) for u_users_items in u_users_items_list])
                u_users_items = torch.tensor([np.pad(item, [[0, max_len - len(item)], [0, 0]]) for item in u_users_items_list]).unsqueeze(0)[torch.zeros(item_id_batch_size, dtype=torch.long), ...].to(device)

                for item_ids_batch in item_ids_batches:
                    iids = item_ids_batch.to(device)
                    length = len(item_ids_batch)
                    i_users_list = i_users[iids, :].to(device)
                    pred_user.append(model(uids[:length], iids, u_items[:length], u_users[:length], u_users_items[:length], i_users_list).cpu())
                    torch.cuda.empty_cache()

                preds.append(np.concatenate(pred_user))

                # for item_id in item_ids:
                #     uid = torch.tensor(user_id).to(device).unsqueeze(0)
                #     iid = torch.tensor(item_id).to(device).unsqueeze(0)
                #     u_items = torch.tensor(valid_data.u_items_list[user_id]).to(device).unsqueeze(0)
                #     u_users = torch.tensor(valid_data.u_users_list[user_id]).to(device).unsqueeze(0)
                #     u_users_items = torch.tensor(valid_data.u_users_items_list[user_id]).to(device).unsqueeze(0)
                #     i_users = torch.tensor(valid_data.i_users_list[item_id]).to(device).unsqueeze(0)
                #     preds.append(model(uid, iid, u_items, u_users, u_users_items, i_users))
            preds = torch.tensor(preds).cpu().squeeze(-1)
            np.save(os.path.join(args.dataset_path, 'preds.npy'), preds)
            precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(torch.tensor(preds).cpu(), train_data.user_dict, valid_data.user_dict, user_ids, item_ids.cpu(), K)
            for k in range(len(K)):
                precision_k.append(np.mean(precision_batch[k]))
                recall_k.append(np.mean(recall_batch[k]))
                ndcg_k.append(np.mean(ndcg_batch[k]))
            # precision_batch = np.mean(precision_batch, axis=1)
            # recall_batch = np.mean(recall_batch, axis=1)
            # ndcg_batch = np.mean(ndcg_batch, axis=1)
        return precision_k, recall_k, ndcg_k
    else:
        return

def validate_by_preds(preds, train_data=None, valid_data=None, user_count=None, item_count=None, K=20):
    precision_k = []
    recall_k = []
    ndcg_k = []
    item_ids = torch.arange(item_count, dtype=torch.long)
    user_ids = list(valid_data.user_dict.keys())
    precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(torch.tensor(preds), train_data.user_dict, valid_data.user_dict, user_ids, item_ids, K)
    for k in range(len(K)):
        precision_k.append(np.mean(precision_batch[k]))
        recall_k.append(np.mean(recall_batch[k]))
        ndcg_k.append(np.mean(ndcg_batch[k]))
    return precision_k, recall_k, ndcg_k

if __name__ == '__main__':
    main()

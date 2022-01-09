import os
import random
import collections
import dgl
import torch
import numpy as np
import pandas as pd
import pickle
from gensim.models import KeyedVectors

class DataLoaderTKGAT(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.name = 'DataLoaderTKGAT_' + args.data_name + '_dump.pkl'
        self.dump_path = os.path.join(args.dump_dir, self.name)
        self.id_offset = {} # entity ID offset dict
        self.id_offset['item'] = 0

        # detect dump file to accelerate process
        if os.path.exists(self.dump_path):
            with open(self.dump_path, 'rb') as dump_file:
                state_dict = pickle.load(dump_file)
            self.__dict__.update(state_dict)
        else:
            self.pretrain_embedding_dir = args.pretrain_embedding_dir
            data_dir = os.path.join(args.data_dir, args.data_name)
            train_file = os.path.join(data_dir, 'train.txt')
            valid_file = os.path.join(data_dir, 'valid.txt')
            test_file = os.path.join(data_dir, 'test.txt')
            kg_file = os.path.join(data_dir, "kg_final.txt")
            tag_file = os.path.join(data_dir, "tagging.txt")
            tag_description_file = os.path.join(data_dir, "tag_list.txt")

            # load data
            self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
            self.cf_valid_data, self.valid_user_dict = self.load_cf(valid_file)
            self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
            self.statistic_cf()

            self.load_glove()
            kg_data = self.load_kg(kg_file)
            tag_data = self.load_kg(tag_file)
            self.tag_description = self.load_tag(tag_description_file)
            self.construct_data(kg_data, tag_data)
            self.train_graph = self.create_graph(self.kg_train_data, self.n_entities)
            self.valid_graph = self.create_graph(self.kg_valid_data, self.n_entities)
            self.test_graph = self.create_graph(self.kg_test_data, self.n_entities)

            # save dump file
            with open(self.dump_path, 'wb') as dump_file:
                pickle.dump(self.__dict__, dump_file, 2)

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.print_info(logging)
        
        if self.use_pretrain == 1:
            self.load_glove()
            self.construct_embeddings()

    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        # record number of users, items and train/test edges
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_valid_data[0]), max(self.cf_test_data[0])) + 1 # user index start from 0
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_valid_data[1]), max(self.cf_test_data[1])) + 1 # item index start from 0
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_valid = len(self.cf_valid_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def load_tag(self, filename):
        tag_desciption = collections.defaultdict(list)
        with open(filename, mode='r') as f:
            for line in f.readlines()[1:]:
                org_id, tag_id, desciption = line.split('\t')
                for word in desciption.split():
                    if word in self.wordvec:
                        tag_desciption[int(tag_id)].append(word)
        return tag_desciption

    def construct_data(self, kg_data, tag_data):
        # plus inverse kg data
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id (add user entities into KG, index following the last entity of KG)
        kg_data['r'] += 2 # 0th relation is user-item, 1st relation is item-user
        self.n_relations_kg = max(kg_data['r']) + 1 # relation nuumber in KG
        self.n_entities_kg = max(max(kg_data['h']), max(kg_data['t'])) + 1 # total entity number in KG
        self.n_users_entities_kg = self.n_users + self.n_entities_kg

        self.id_offset['word_relations'] = self.n_relations_kg
        self.id_offset['tag'] = self.id_offset['word_relations'] + 4 # user-word, item-word and their reverse, 4 relations in total
        self.id_offset['user'] = self.n_entities_kg
        self.id_offset['word'] = self.n_users_entities_kg
        
        # add tag-related word nodes
        word_entities = set()
        for description in self.tag_description.values():
            for word in description:
                word_entities.add(word)
        self.word_entities = list(word_entities)
        word_entity_id = {entity: id for id, entity in enumerate(self.word_entities)}
        self.n_words = len(self.word_entities) # number of added word nodes

        word_interaction_data = []
        for i, row in tag_data.iterrows():
            user, tag, item = row['h'], row['r'], row['t']
            for word in self.tag_description[tag]:
                word_interaction_data.append((user + self.id_offset['user'], 0 + self.id_offset['word_relations'], word_entity_id[word] + self.id_offset['word']))
                word_interaction_data.append((item + self.id_offset['item'], 1 + self.id_offset['word_relations'], word_entity_id[word] + self.id_offset['word']))
        word_data = pd.DataFrame(data=word_interaction_data, columns=['h', 'r', 't']).drop_duplicates()
        reverse_word_data = word_data.copy()
        reverse_word_data = reverse_word_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_word_data['r'] += 2
        kg_data = pd.concat([kg_data, word_data, reverse_word_data], axis=0, ignore_index=True, sort=False)

        # add tag interactions
        self.n_tags = max(tag_data['r']) + 1
        tag_data['r'] += self.id_offset['tag']
        reverse_tag_data = tag_data.copy()
        reverse_tag_data = reverse_tag_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_tag_data['r'] += self.n_tags
        kg_data = pd.concat([kg_data, tag_data, reverse_tag_data], axis=0, ignore_index=True, sort=False)

        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_relations = max(kg_data['r']) + 1
        

        # prepare user-item interaction data
        self.cf_train_data = (np.array(list(map(lambda d: d + self.id_offset['user'], self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_valid_data = (np.array(list(map(lambda d: d + self.id_offset['user'], self.cf_valid_data[0]))).astype(np.int32), self.cf_valid_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.id_offset['user'], self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.id_offset['user']: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.valid_user_dict = {k + self.id_offset['user']: np.unique(v).astype(np.int32) for k, v in self.valid_user_dict.items()}
        self.test_user_dict = {k + self.id_offset['user']: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        # 0: user-item interaction relation
        # 1: the reverse of the above
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        cf2kg_valid_data = pd.DataFrame(np.zeros((self.n_cf_valid, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_valid_data['h'] = self.cf_valid_data[0]
        cf2kg_valid_data['t'] = self.cf_valid_data[1]
        reverse_cf2kg_valid_data = pd.DataFrame(np.ones((self.n_cf_valid, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_valid_data['h'] = self.cf_valid_data[1]
        reverse_cf2kg_valid_data['t'] = self.cf_valid_data[0]

        cf2kg_test_data = pd.DataFrame(np.zeros((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_test_data['h'] = self.cf_test_data[0]
        cf2kg_test_data['t'] = self.cf_test_data[1]
        reverse_cf2kg_test_data = pd.DataFrame(np.ones((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_test_data['h'] = self.cf_test_data[1]
        reverse_cf2kg_test_data['t'] = self.cf_test_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)
        self.kg_valid_data = pd.concat([kg_data, cf2kg_valid_data, reverse_cf2kg_valid_data], ignore_index=True)
        self.kg_test_data = pd.concat([kg_data, cf2kg_test_data, reverse_cf2kg_test_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_valid = len(self.kg_valid_data)
        self.n_kg_test = len(self.kg_test_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
            
        self.valid_kg_dict = collections.defaultdict(list)
        self.valid_relation_dict = collections.defaultdict(list)
        for row in self.kg_valid_data.iterrows():
            h, r, t = row[1]
            self.valid_kg_dict[h].append((t, r))
            self.valid_relation_dict[r].append((h, t))

        self.test_kg_dict = collections.defaultdict(list)
        self.test_relation_dict = collections.defaultdict(list)
        for row in self.kg_test_data.iterrows():
            h, r, t = row[1]
            self.test_kg_dict[h].append((t, r))
            self.test_relation_dict[r].append((h, t))


    def print_info(self, logging):
        logging.info('n_users:                   %d' % self.n_users)
        logging.info('n_items:                   %d' % self.n_items)
        logging.info('n_entities_kg:             %d' % self.n_entities_kg)          # number of entities in KG
        logging.info('n_users_entities_kg:       %d' % self.n_users_entities_kg)
        logging.info('n_relations_kg:            %d' % self.n_relations_kg)         # number of relations in KG
        logging.info('n_tags:                    %d' % self.n_tags)                 # number of tags
        logging.info('n_words:                   %d' % self.n_words)                # number of added word nodes
        logging.info('n_entities:                %d' % self.n_entities)             # number of total entities (include item, KG entities, user, word nodes)
        logging.info('n_relations:               %d' % self.n_relations)            # number of total relations (include KG relations, interaction relations)
        logging.info('n_cf_train:                %d' % self.n_cf_train)
        logging.info('n_cf_valid:                %d' % self.n_cf_valid)
        logging.info('n_cf_test:                 %d' % self.n_cf_test)
        logging.info('n_kg_train:                %d' % self.n_kg_train)
        logging.info('n_kg_valid:                %d' % self.n_kg_valid)
        logging.info('n_kg_test:                 %d' % self.n_kg_test)


    def create_graph(self, kg_data, n_nodes):
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(kg_data['t'], kg_data['h']) # DGL aggregates message into tail node, so the edges need to be reversed
        g.readonly()
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        g.edata['type'] = torch.LongTensor(kg_data['r'])
        return g


    # def load_pretrain_wordvec(self, filename):
    #     print("Loading Glove Model")
    #     glove_model = {}
    #     with open(File,'r') as f:
    #         for line in f:
    #             split_line = line.split()
    #             word = split_line[0]
    #             embedding = np.array(split_line[1:], dtype=np.float64)
    #             glove_model[word] = embedding
    #     print(f"{len(glove_model)} words loaded!")
    #     return glove_model


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items, att_score): # TODO: active sampling
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            # neg_item_id = np.random.choice(np.arange(self.n_items), p=1-att_score)
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items


    def generate_cf_batch(self, user_dict, att_score=None):
        exist_users = user_dict.keys()
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(self.cf_batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1, att_score)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=self.n_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.entity_dim
        assert self.item_pre_embed.shape[1] == self.args.entity_dim

    def load_glove(self):
        wordvec_path = os.path.join(self.args.data_dir, 'pretrain/GloVe.kv')
        self.wordvec = KeyedVectors.load(wordvec_path)
        assert self.wordvec.vector_size == self.args.entity_dim

    def construct_embeddings(self):
        # word embeddings
        self.word_embed = np.zeros([self.n_words, self.args.entity_dim], dtype=np.float32)
        for i, word in enumerate(self.word_entities):
            self.word_embed[i, :] = self.wordvec[word]

        # tag embeddings
        self.tag_embed = np.zeros([self.n_tags, self.args.entity_dim], dtype=np.float32)
        for i in range(self.n_tags):
            if len(self.tag_description[i]) == 0:
                self.tag_embed[i, :] = np.random.uniform(low=-1.0, high=1.0, size=self.args.entity_dim)
            else:
                self.tag_embed[i, :] = np.asarray([self.wordvec[word] for word in self.tag_description[i]]).mean(axis=0)







from platform import node
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from utility.helper import edge_softmax_fix


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, n_attention_heads=1, device='cpu'):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.n_attention_heads = n_attention_heads
        assert(self.out_dim % self.n_attention_heads == 0)
        self.message_dropout = nn.Dropout(dropout)
        self.node_batch_size = 1000
        self.device = device

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim//self.n_attention_heads)       # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim//self.n_attention_heads)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim//self.n_attention_heads)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim//self.n_attention_heads)      # W2 in Equation (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()


    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed.unsqueeze(-2)            # (n_nodes, 1, in_dim)
        
        # Equation (3) & (10)
        # DGL: dgl-cu90(0.4.1)
        # Get different results when using `dgl.function.sum`, and the randomness is due to `atomicAdd`
        # Use `dgl.function.sum` when training model to speed up
        # Use custom function to ensure deterministic behavior when predicting

        # att: (n_attention_heads, 1)
        # side/N_h: (n_attention_heads, in_dim)
        
        # N_h_full = []
        # node_idx = torch.arange(g.num_nodes()).to(self.device)
        # node_batches = [node_idx[i:i+self.node_batch_size] for i in range(0, g.num_nodes(), self.node_batch_size)]
        # for node_batch in node_batches:
        #     sub_g = g.subgraph(node_batch)
        #     # if mode == 'predict':
        #     #     sub_g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        #     # else:
        #     sub_g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))
        #     N_h_full.append(sub_g.ndata['N_h'])
        
        # g.ndata['N_h'] = torch.cat(N_h_full)

        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            # print('g.ndata["node"].shape = {}'.format(g.ndata['node'].shape))
            # print('g.ndata["N_h"].shape = {}'.format(g.ndata['N_h'].shape))
            out = self.activation(self.W((g.ndata['node'] + g.ndata['N_h']).reshape([-1, self.in_dim]))).reshape([-1, self.out_dim])                         # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))      # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1((g.ndata['node'] + g.ndata['N_h']).reshape([-1, self.in_dim]))).reshape([-1, self.out_dim])                       # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2((g.ndata['node'] * g.ndata['N_h']).reshape([-1, self.in_dim]))).reshape([-1, self.out_dim])                       # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out


class TKGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 word_embed=None, tag_embed=None, word_offset=None, tag_offset=None, device='cpu'):

        super(TKGAT, self).__init__()
        self.use_pretrain = args.use_pretrain
        self.device = device
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.word_offset = word_offset
        self.tag_offset = tag_offset

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.n_attention_heads = args.n_attention_heads
        self.attention_dim = args.attention_dim

        for dim in self.conv_dim_list[1:]:
            if dim % self.n_attention_heads != 0:
                raise RuntimeError(f'Dimension (got {dim}) should be divisible by number of heads (got {self.n_attention_heads}).')

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities, self.entity_dim)
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        self.W_A = nn.Parameter(torch.Tensor(self.relation_dim, self.n_attention_heads, self.attention_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_A, gain=nn.init.calculate_gain('relu'))

        if (self.use_pretrain == 1) and (word_embed is not None) and (tag_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - word_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([other_entity_embed, word_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

            other_relation_embed = nn.Parameter(torch.Tensor(self.n_relations - tag_embed.shape[0], self.relation_dim))
            nn.init.xavier_uniform_(other_relation_embed, gain=nn.init.calculate_gain('relu'))
            tag_embed_trans = torch.bmm(tag_embed.unsqueeze(1), self.W_R.data[-tag_embed.shape[0]:, ...]).squeeze(1)
            relation_embed = torch.cat([other_relation_embed, tag_embed_trans], dim=0)
            self.relation_embed.weight = nn.Parameter(relation_embed)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type, self.n_attention_heads, device=self.device))


    def att_score(self, edges):
        # Equation (4)
        W_ra = torch.matmul(self.W_r, self.W_A.reshape([self.relation_dim, -1]))                      # (entity_dim, n_attention_heads*attention_dim)
        r_mul_t = torch.tanh(torch.matmul(self.entity_user_embed(edges.src['id']), W_ra))                         # (n_edge, n_attention_heads*attention_dim)
        r_embed = self.relation_embed(edges.data['type'])                                             # (1, relation_dim)
        r_embed_a = torch.matmul(r_embed, self.W_A.reshape([self.relation_dim, -1]))                  # (1, n_attention_heads*attention_dim)
        r_mul_h = torch.tanh(torch.matmul(self.entity_user_embed(edges.dst['id']), W_ra) + r_embed_a)           # (n_edge, n_attention_heads*attention_dim)
        att = r_mul_t*r_mul_h
        del r_mul_h, r_mul_t, W_ra, r_embed, r_embed_a
        att = torch.sum(att.reshape([-1, self.n_attention_heads, self.attention_dim]), dim=-1).reshape([-1, self.n_attention_heads, 1])              # (n_edge, n_attention_heads, 1)
        return {'att': att}


    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self.att_score, edge_idxs)

        # Equation (5)
        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                 # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entity_user_embed(h)              # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_user_embed(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_user_embed(neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def cf_embedding(self, mode, g):
        g = g.local_var()
        ego_embed = self.entity_user_embed(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, cf_concat_dim)
        return all_embed


    def cf_score(self, mode, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        all_embed = self.cf_embedding(mode, g)          # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[item_ids]                # (n_eval_items, cf_concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_eval_users, n_eval_items)
        return cf_score


    def calc_cf_loss(self, mode, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.cf_embedding(mode, g)                      # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, cf_concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, cf_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, cf_concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)



import torch
import torch.nn as nn
import torch.nn.functional as F

from kgat.config import KGATConfig
from kgat.core.aggregator import Aggregator


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.0)


class KGAT(nn.Module):
    def __init__(
        self,
        config: KGATConfig,
        n_users,
        n_entities,
        n_relations,
        A_in=None,
        user_pre_embed=None,
        item_pre_embed=None,
    ) -> None:
        super(KGAT, self).__init__()

        self.use_pretrain = config.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = config.embed_dim
        self.relation_dim = config.relation_dim

        self.aggregation_type = config.aggregation_type
        self.conv_dim_list = (
            config.conv_dim_list
            if isinstance(config.conv_dim_list, (list, tuple))
            else eval(config.conv_dim_list)
        )
        self.mess_dropout = (
            config.mess_dropout
            if isinstance(config.mess_dropout, (list, tuple))
            else eval(config.mess_dropout)
        )
        self.n_layers = len(self.conv_dim_list) - 1

        self.kg_l2loss_lambda = config.kg_l2loss_lambda
        self.cf_l2loss_lambda = config.cf_l2loss_lambda

        self.entity_user_embed = nn.Embedding(
            self.n_entities + self.n_users, self.embed_dim
        )
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(
            torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim)
        )

        if (
            (self.use_pretrain == 1)
            and (user_pre_embed is not None)
            and (item_pre_embed is not None)
        ):
            other_entity_embed = nn.Parameter(
                torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim)
            )
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat(
                [item_pre_embed, other_entity_embed, user_pre_embed], dim=0
            )
            self.entity_user_embed = nn.Embedding.from_pretrained(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(
                    self.conv_dim_list[k],
                    self.conv_dim_list[k + 1],
                    self.mess_dropout[k],
                    self.aggregation_type,
                )
            )

        # 空のスパーステンソルを作成
        size = (self.n_users + self.n_entities, self.n_users + self.n_entities)
        indices = torch.zeros((2, 0), dtype=torch.long)  # 空のindices
        values = torch.zeros(0)  # 空のvalues
        empty_sparse = torch.sparse_coo_tensor(indices, values, size)

        self.A_in = nn.Parameter(empty_sparse)

        # A_inが渡された場合は上書き
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)
        return all_embed

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """

        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = (
            _L2_loss_mean(user_embed)
            + _L2_loss_mean(item_pos_embed)
            + _L2_loss_mean(item_neg_embed)
        )

        l2_loss = (
            _L2_loss_mean(user_embed)
            + _L2_loss_mean(item_pos_embed)
            + _L2_loss_mean(item_neg_embed)
        )
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """

        r_embed = self.relation_embed(r)

        W_r = self.trans_M[r]

        h_embed = self.entity_user_embed(h)
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        kg_loss = (-1) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = (
            _L2_loss_mean(r_mul_h)
            + _L2_loss_mean(r_embed)
            + _L2_loss_mean(r_mul_pos_t)
            + _L2_loss_mean(r_mul_neg_t)
        )
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_likst, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_likst]
        t_embed = self.entity_user_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations) -> None:
        device = self.A_in.device

        rows = []
        cols = []
        vals = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(
                batch_h_list, batch_t_list, r_idx
            )
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            vals.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse_coo_tensor(indices, vals, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """

        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_embed = all_embed[item_ids]

        scores = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return scores

    def forward(self, *input: torch.Tensor, mode: str):
        if mode == "train_cf":
            return self.calc_cf_loss(*input)
        elif mode == "train_kg":
            return self.calc_kg_loss(*input)
        elif mode == "update_att":
            return self.update_attention(*input)
        elif mode == "predict":
            return self.calc_score(*input)

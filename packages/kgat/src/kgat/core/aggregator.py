import torch
import torch.nn as nn

from kgat.config import KGATConfig


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type) -> None:
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == "gcn":
            self.linear = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.aggregator_type == "graphsage":
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.aggregator_type == "bi-interaction":
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError(
                "Aggregator type {} not recognized.".format(self.aggregator_type)
            )

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """

        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == "gcn":
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))
            embeddings = self.message_dropout(embeddings)
            return embeddings
        elif self.aggregator_type == "graphsage":
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))
            embeddings = self.message_dropout(embeddings)
            return embeddings
        elif self.aggregator_type == "bi-interaction":
            sum_embeddings = self.activation(
                self.linear1(ego_embeddings + side_embeddings)
            )
            bi_embeddings = self.activation(
                self.linear2(ego_embeddings * side_embeddings)
            )
            embeddings = bi_embeddings + sum_embeddings
            embeddings = self.message_dropout(embeddings)
            return embeddings
        else:
            raise NotImplementedError(
                "Aggregator type {} not recognized.".format(self.aggregator_type)
            )

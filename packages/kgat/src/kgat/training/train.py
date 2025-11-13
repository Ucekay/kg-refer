import logging
import random
import sys
from time import time

import numpy as np
import pandas as pd
import torch
from torch._dynamo.trace_rules import torch_c_binding_in_graph_functions

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat.training.evaluate import evaluate
from kgat.utils.log_helper import create_log_id, logging_config
from kgat.utils.model_helper import early_stopping, load_model, save_model


def train(config: KGATConfig):
    random.seed(config.seed)
    # np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    save_dir = "packages/kgat/trained_model/KGAT/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/".format(
        config.data_name,
        config.embed_dim,
        config.relation_dim,
        config.laplacian_type,
        config.aggregation_type,
        "-".join(
            [
                str(i)
                for i in (
                    config.conv_dim_list
                    if isinstance(config.conv_dim_list, (list, tuple))
                    else eval(config.conv_dim_list)
                )
            ]
        ),
        config.lr,
        config.use_pretrain,
    )
    log_save_id = create_log_id(save_dir)
    logging_config(folder=save_dir, name=f"log{log_save_id}", no_console=False)
    logger = logging.getLogger(__name__)
    logger.info(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoader(config, logger)
    if config.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed = None
        item_pre_embed = None

    model = KGAT(
        config,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
        user_pre_embed,
        item_pre_embed,
    )
    if config.use_pretrain == 2:
        model = load_model(model, config.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    kg_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_epoch = -1
    best_recall = 0

    Ks: list[int] = eval(config.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {"precision": [], "recall": [], "ndcg": []} for k in Ks}

    for epoch in range(1, config.n_epoch + 1):
        time0 = time()
        model.train()
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = (
                data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            )
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode="train_cf"
            )

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info(
                    f"Error (CF Training): Epoch {epoch:d} Iter {iter:d} / {n_cf_batch:d} - Loss is NaN"
                )
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % config.cf_print_every) == 0:
                logging.info(
                    f"CF Training: Epoch {epoch:04d} Total Iter {iter:04d} | Total Time {time() - time1:1f}s | Iter Mean Loss {cf_total_loss / n_cf_batch:.4f}"
                )

        logging.info(
            f"CF Training: Epoch {epoch:04d} Total Iter {n_cf_batch:04d} | Total Time {time() - time1:1f} | Iter Mean Loss {cf_total_loss / n_cf_batch:.4f}"
        )

        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = (
                data.generate_kg_batch(
                    data.train_kg_dict, data.kg_batch_size, data.n_users_entities
                )
            )
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(
                kg_batch_head,
                kg_batch_relation,
                kg_batch_pos_tail,
                kg_batch_neg_tail,
                mode="train_kg",
            )

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    f"Error (KG Training): Epoch {epoch:04d} Iter {iter:04d} / {n_kg_batch:04d} - Loss is NaN"
                )
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % config.kg_print_every) == 0:
                logging.info(
                    f"KG Training: Epoch {epoch:04d} Total Iter {iter:04d} | Iter Time {time() - time4:1f}s | Iter Mean Loss {kg_total_loss / n_kg_batch:.4f}"
                )

        logging.info(
            f"KG Training: Epoch {epoch:04d} Total Iter {n_kg_batch:04d} | Total Time {time() - time3:1f}s | Iter Mean Loss {kg_total_loss / n_kg_batch:.4f}"
        )

        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode="update_att")
        logging.info(
            f"Update Attention: Epoch {epoch:04d} | Total Time {time() - time5:1f}s"
        )

        logging.info(
            f"CF + KG Training: Epoch {epoch:04d} | Total Time {time() - time0:1f}s"
        )

        if (epoch % config.evaluate_every) == 0 or epoch == config.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info(
                f"CF Evaluation: Epoch {epoch:04d} | Total time {time() - time6:1f}s | Precision [{metrics_dict[k_min]['precision']}, {metrics_dict[k_max]['precision']}], Recall [{metrics_dict[k_min]['recall']}, {metrics_dict[k_max]['recall']}], NDCG [{metrics_dict[k_min]['ndcg']}, {metrics_dict[k_max]['ndcg']}]"
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ["precision", "recall", "ndcg"]:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(
                metrics_list[k_min]["recall"], config.stopping_steps
            )

            if should_stop:
                break
            if metrics_list[k_min]["recall"].index(best_recall) == len(epoch_list) - 1:
                save_model(model, save_dir, epoch, best_epoch)
                logging.info(f"Save model on epoch {epoch:d}")
                best_epoch = epoch

    metrics_df = [epoch_list]
    metrics_cols = ["epoch_idx"]
    for k in Ks:
        for m in ["precision", "recall", "ndcg"]:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append(f"{m}@{k}")
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(save_dir + "/metrics.csv", sep="\t", index=False)

    best_metrics = (
        metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
    )
    logging.info(
        "Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
            int(best_metrics["epoch_idx"]),
            best_metrics["precision@{}".format(k_min)],
            best_metrics["precision@{}".format(k_max)],
            best_metrics["recall@{}".format(k_min)],
            best_metrics["recall@{}".format(k_max)],
            best_metrics["ndcg@{}".format(k_min)],
            best_metrics["ndcg@{}".format(k_max)],
        )
    )

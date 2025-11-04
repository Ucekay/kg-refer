from sentence_transformers import SentenceTransformer
from torch.types import Tensor
from torchvision.io.video import np

from edc.utils import llm_utils


class SchemaRetriever:
    def __init__(
        self,
        target_schema_dict: dict[str, str],
        embedding_model: SentenceTransformer,
        embedding_tokenizer,
        finetuned_e5mistral=False,
    ) -> None:
        self.target_schema_dict = target_schema_dict
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer

        self.target_schema_embedding_dict: dict[str, Tensor] = {}
        self.finetuned_e5mistral = finetuned_e5mistral

        for relation, relation_definition in target_schema_dict.items():
            if finetuned_e5mistral:
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model, self.embedding_tokenizer, relation_definition
                )
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    relation_definition,
                    prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
                )
            self.target_schema_embedding_dict[relation] = embedding

    def update_schema_embedding_dict(self):
        for relation, relation_definition in self.target_schema_dict.items():
            if relation in self.target_schema_embedding_dict:
                continue
            if self.finetuned_e5mistral:
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model, self.embedding_tokenizer, relation_definition
                )
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model, relation_definition
                )
            self.target_schema_embedding_dict[relation] = embedding

    def retrieve_relevant_relations(self, query_input_text: str, top_k=10):
        target_relation_list = list(self.target_schema_embedding_dict.keys())
        target_relation_embedding_list = list(
            self.target_schema_embedding_dict.values()
        )

        if self.finetuned_e5mistral:
            query_embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                query_input_text,
                "Retrieve descriptions of relations that are present in the given text.",
            )
        else:
            query_embedding = llm_utils.get_embedding_sts(
                self.embedding_model,
                query_input_text,
                prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
            )

        scores = (
            np.array([query_embedding]) @ np.array(target_relation_embedding_list).T
        )

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return [target_relation_list[i] for i in highest_score_indices[:top_k]]

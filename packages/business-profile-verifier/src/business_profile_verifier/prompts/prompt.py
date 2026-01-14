"""Prompt formatting module for verification."""

import torch


class PromptCausalLM:
    """Handle chat template formatting for causal language models."""

    def __init__(self, model_path, tokenizer, max_length, system_message=None):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def set_system_message(self, system_message):
        self.system_message = system_message

    def _get_system_prompt(self, conversation_history):
        if "gemma" in self.model_path.lower():
            assert conversation_history[0]["role"] == "user"
            old_content = conversation_history[0]["content"]
            conversation_history[0]["content"] = """{system_prompt}
{old_content}""".format(
                system_prompt=self.system_message,
                old_content=old_content
            )
        else:
            conversation_history = [
                {
                    "role": "system",
                    "content": self.system_message
                },
                *conversation_history
            ]
        return conversation_history

    def _get_inputs(self, messages, response=None, add_generation_prompt=False):
        conversation_history = []
        if type(messages) in (list, tuple):
            conversation_history.extend(messages)
        else:
            conversation_history.append(messages)
        conversation_history = self._get_system_prompt(conversation_history)
        input_ids = self.tokenizer.apply_chat_template(
            conversation=conversation_history,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            truncation=False,
            padding="do_not_pad"
        )
        labels_start = 0
        if response is not None:
            labels_start = len(input_ids)
            input_ids.extend(self.tokenizer.encode(
                response,
                add_special_tokens=False,
                truncation=False,
                padding="do_not_pad"
            ))
            input_ids.append(self.tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()
        labels[:labels_start] = -100
        input_ids = torch.narrow(input_ids, 0, 0, min(input_ids.shape[-1], self.max_length))
        labels = torch.narrow(labels, 0, 0, min(labels.shape[-1], self.max_length))
        return input_ids, labels

    def get_inputs(self, messages, response=None, add_generation_prompt=False):
        if type(messages) in (list, tuple) and type(messages[0]) in (list, tuple):
            batched_input_ids, batched_labels = self._get_batch_inputs(messages, response)
        else:
            input_ids, labels = self._get_inputs(messages, response, add_generation_prompt)
            batched_input_ids, batched_labels = (input_ids,), (labels,)
        return torch.stack(batched_input_ids), torch.stack(batched_labels)

    def _get_batch_inputs(self, messages, response=None, add_generation_prompt=True):
        if response is None:
            response = [None] * len(messages)
        assert len(response) == len(messages)

        def _get_inputs_wrapper(messages, response):
            return self._get_inputs(messages, response, add_generation_prompt=add_generation_prompt)

        batch_inputs = list(map(_get_inputs_wrapper, *zip(*zip(messages, response))))
        max_length = max(inp.size(0) for (inp, _) in batch_inputs)
        batch_input_ids, batch_labels = [], []
        for (inp, labels) in batch_inputs:
            # Pad to left
            batch_input_ids.append(torch.cat((torch.full((max_length - inp.size(0),), self.tokenizer.pad_token_id), inp)))
            batch_labels.append(torch.cat((torch.full((max_length - labels.size(0),), self.tokenizer.pad_token_id), labels)))
        return batch_input_ids, batch_labels

    def extract_response_ids(self, outputs, input_ids, num_return_sequences=1):
        assert len(outputs) == (len(input_ids) * num_return_sequences)
        response_ids = []
        for inp_idx in range(len(input_ids)):
            for seq_idx in range(num_return_sequences):
                response_ids.append(outputs[seq_idx + inp_idx * num_return_sequences, len(input_ids[inp_idx]):])
        return response_ids

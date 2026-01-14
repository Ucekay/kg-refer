"""Business Profile Verifier module."""

import json
import logging
import os
import re
from typing import Any, Dict, List

import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import PromptCausalLM


SYSTEM_MESSAGE = """You are verifying how factual a response is by extracting fine-grained, verifiable claims. Each claim must describe one single event or one single state in one sentence.

CRITICAL: The provided evidence is a Business Profile that describes what the business ACTUALLY OFFERS. The Business Profile uses phrases like "Users who enjoy X" or "Fans of Y" to describe the business's actual features and offerings.

HOW TO EXTRACT CLAIMS FROM THE RESPONSE:
Break down the response into individual factual claims about what the business offers or provides.
- Each claim should describe ONE feature or characteristic
- Remove phrases like "The user would enjoy" - focus on what the BUSINESS HAS
- Convert user-focused statements to business-focused statements

EXAMPLES:
1. Response: "The user would enjoy Tony's Pizza because it offers authentic Italian dishes and crispy crusts in a cozy setting."
   Extract claims:
   - Tony's Pizza offers authentic Italian dishes.
   - Tony's Pizza has crispy crusts.
   - Tony's Pizza provides a cozy setting.

2. Response: "The user would enjoy The Local Pub for its craft beer selection and friendly atmosphere."
   Extract claims:
   - The Local Pub serves craft beer.
   - The Local Pub has a friendly atmosphere.

RULES FOR INTERPRETING BUSINESS PROFILE:
- "Users who would enjoy [feature]" → The business HAS this feature
- "Fans of [food/drink]" → The business OFFERS this food/drink
- "[Food] enthusiasts looking for [quality]" → The business SERVES this food with these qualities

3. Business Profile: "Pizza enthusiasts looking for authentic Italian dishes and crispy crusts would enjoy Tony's Pizza"
   This means Tony's Pizza ACTUALLY OFFERS:
   - authentic Italian dishes
   - crispy crusts

4. Business Profile: "Fans of craft beer and those looking for a cozy atmosphere would enjoy The Local Pub"
   This means The Local Pub ACTUALLY OFFERS:
   - craft beer
   - a cozy atmosphere

For each extracted claim, classify it as follows:

Supported: The claim describes a feature that the Business Profile states the business has.

Unsupported: The claim describes something NOT mentioned in or contradicted by the Business Profile.

Output format:
<fact 1>: <your judgment of fact 1>
<fact 2>: <your judgment of fact 2>
...
<fact n>: <your judgment of fact n>

If no verifiable claim can be extracted, simply output "No verifiable claim."
"""


def extract_response_ids(outputs, input_ids, num_return_sequences=1):
    """Extract response IDs from model outputs."""
    assert len(outputs) == (len(input_ids) * num_return_sequences)
    response_ids = []
    for inp_idx in range(len(input_ids)):
        for seq_idx in range(num_return_sequences):
            response_ids.append(outputs[seq_idx + inp_idx * num_return_sequences, len(input_ids[inp_idx]):])
    return response_ids


def extract_business_profile(input_str: str) -> str:
    """Extract Business profile from input_str.

    Args:
        input_str: The input string containing Business profile.

    Returns:
        The extracted Business profile text.
    """
    # Match from "Business profile:" until "User profile:" or end of string
    pattern = r"Business profile:\s*(.*?)(?=User profile:|### For the given|\Z)"
    match = re.search(pattern, input_str, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class BusinessProfileVerifier:
    """Verify explanation outputs using Business profile as source."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./data/cache",
        output_dir: str = "./output",
    ):
        """Initialize the verifier.

        Args:
            model_name: Name or path of the model to use.
            cache_dir: Directory for caching.
            output_dir: Directory for output files.
        """
        self.median_claims = 17
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = 2048
        self.num_beams = 1
        self.do_sample = True
        self.temperature = 0.7
        self.top_p = 1
        self.num_return_sequences = 1
        self.max_length = 108000
        self.system_message = SYSTEM_MESSAGE
        self.label_n = 2

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.output_dir = output_dir
        self.cache_dir = cache_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            raise

        self.prompter = PromptCausalLM(
            self.model_name,
            self.tokenizer,
            self.max_length,
            self.system_message
        )

    def get_prompt(self, role: str, **args) -> str:
        """Generate prompt for the given role."""
        prompt = ""
        if role == "user":
            prompt += "### Response\n" + args["response"]
            prompt += "\n" + "### Evidence\n"
            prompt += args["evidence"]
        elif role == "assistant":
            prompt += "### Facts\n"
            if args.get("claim_verification_result") is None or len(args["claim_verification_result"]) == 0:
                prompt += "No verifiable claim."
                return prompt
            prompt_claims = []
            for claim in args["claim_verification_result"]:
                label_text = "Supported" if claim["verification_result"].lower() == "supported" else "Unsupported"
                prompt_claims.append("{}: {}".format(claim["claim"], label_text))
            prompt += "\n".join(prompt_claims)
        else:
            raise ValueError("Unrecognized role: {}".format(role))
        return prompt

    def _extract_judgment(self, response):
        """Extract judgments from model response."""
        judgmentPattern = r"(?P<claim_text>.+): (?P<judgment>Supported|Unsupported)\n?"
        return re.findall(judgmentPattern, response)

    def verify(self, data: List[Dict[str, Any]], input_file_name: str) -> Dict[str, List[float]]:
        """Verify explanations using Business profile as evidence."""
        time_taken = {
            "evidence_extraction": [],
            "decompose_and_verify": [],
            "verifastscore": []
        }

        processed_data = []
        for dict_item in tqdm(data, desc="Extracting Business profile evidence"):
            input_str = dict_item.get("input_str", "")
            output_str = dict_item.get("output_str", "")

            business_profile = extract_business_profile(input_str)
            dict_item["business_profile"] = business_profile
            dict_item["evidence"] = business_profile
            processed_data.append(dict_item)

        output_dir = os.path.join(self.output_dir, "model_output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = "verification_{}_{}.jsonl".format(input_file_name, self.label_n)
        output_path = os.path.join(output_dir, output_file)

        verifastscore = []
        with open(output_path, "w") as f:
            for dict_item in tqdm(processed_data, desc="Verifying claims"):
                output_str = dict_item.get("output_str", "").strip()
                evidence = dict_item.get("evidence", "").strip()

                if not output_str:
                    dict_item["claim_verification_result"] = []
                    f.write(json.dumps(dict_item) + "\n")
                    continue

                messages = [{
                    "role": "user",
                    "content": self.get_prompt(
                        role="user",
                        response=output_str,
                        evidence=evidence
                    )
                }]

                input_ids, _ = self.prompter.get_inputs(messages)
                outputs = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=torch.ones_like(input_ids).to(self.device),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_return_sequences=self.num_return_sequences
                )
                response_ids = extract_response_ids(
                    outputs,
                    input_ids=input_ids,
                    num_return_sequences=self.num_return_sequences
                )
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=False)
                logging.info("Input:\n{}\n\nOutput:\n{}\n****".format(messages[0]["content"], responses[0]))
                prediction = self._extract_judgment(responses[0])

                if len(prediction) == 0 or "No verifiable claim".lower() in prediction[0][0].lower():
                    verifastscore.append(0)
                    dict_item["claim_verification_result"] = []
                    f.write(json.dumps(dict_item) + "\n")
                    continue

                fp = [1 if pred_label[1].lower() == "supported" else 0 for pred_label in prediction]
                precision = sum(fp) / len(fp) if len(fp) else 0
                recall = min(1, (len(fp) / self.median_claims))
                score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                verifastscore.append(score)

                claim_verify_res_dict = []
                for pred in prediction:
                    claim_verify_res_dict.append({
                        "claim": pred[0],
                        "verification_result": pred[1].lower()
                    })
                dict_item["claim_verification_result"] = claim_verify_res_dict
                f.write(json.dumps(dict_item) + "\n")

        if len(verifastscore):
            print("VeriFastScore: {:0.4f} ({} instances)".format(sum(verifastscore) / len(verifastscore), len(verifastscore)))

        return time_taken

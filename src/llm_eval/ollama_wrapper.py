"""Wrapper for Ollama chat completions and script to run experiments.

Usage:
    python -m code.llm_eval.ollama_wrapper --models qwen3_32b deepseek_r1 qwen3_32b_think deepseek_r1_think llama3.3_70b llama3.2 mistral --perturbations numeration approximation negative_number random_replacement range masking --host http://localhost:11435 --shot neg-shot --labels true false
    python -m code.llm_eval.ollama_wrapper --models qwen3_32b deepseek_r1 qwen3_32b_think deepseek_r1_think llama3.3_70b llama3.2 mistral --perturbations numeration approximation negative_number random_replacement range masking --host http://localhost:11434 --shot 0-shot --labels true false
"""  # noqa E501

import json
import yaml
from typing import Any, Dict, List
import logging

from ollama import Client, ChatResponse
from ollama._types import ResponseError
import os
from code.prompt import (
    PromptType,
    get_fact_check_system_prompt,
    get_fact_check_user_prompt,
    get_output_schema,
)


from pydantic import BaseModel
import argparse
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class Label(BaseModel):
    label: bool


class OllamaLLM:
    """Wrapper for Ollama chat completions."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        host: str = "https://ollama.ux.uis.no:11434",
    ) -> None:
        """Initializes the OllamaLLM with model configuration.

        Args:
          config_path: Path to the YAML file containing model parameters.

        Raises:
          FileNotFoundError: If the config file does not exist.
          yaml.YAMLError: If the config file cannot be parsed.
        """
        self._config: Dict[str, Any] = self._load_config(config_path)
        self._client = Client(
            host=host,
            headers={"Authorization": f"Bearer {os.getenv('JWT_TOKEN')}"},
        )
        self._model_name = self._config.get("model")
        self._model_options = self._config.get("options", {})
        self._stream = self._config.get("stream", False)
        self._think = self._config.get("think", False)

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Loads YAML configuration from a given file.

        Args:
          path: Path to the YAML configuration file.

        Returns:
          A dictionary of configuration parameters.

        Raises:
          FileNotFoundError: If the file is not found.
          yaml.YAMLError: If the file content is invalid YAML.
        """
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            # Handle empty YAML files (yaml.safe_load returns None)
            return config if config is not None else {}

    def _pull_model(self, model_name: str) -> None:
        """Pulls the specified model from Ollama registry.

        Args:
          model_name: Name of the model to pull.

        Raises:
          Exception: If the model pull fails.
        """
        try:
            print(f"Model '{model_name}' not found. Pulling model...")
            logging.info(f"Pulling model: {model_name}")

            # Pull the model
            self._client.pull(model_name)
            print(f"Successfully pulled model: {model_name}")
            logging.info(f"Successfully pulled model: {model_name}")

        except Exception as e:
            error_msg = f"Failed to pull model '{model_name}': {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def check_model_capabilities(self) -> Dict[str, Any]:
        """Check the capabilities and details of the current model.

        Returns:
          Dictionary containing model information.
        """
        try:
            model_info = self._client.show(self._model_name)
            # Convert ShowResponse to dict
            return (
                model_info.model_dump()
                if hasattr(model_info, "model_dump")
                else dict(model_info)
            )
        except Exception as e:
            logging.warning(f"Could not retrieve model info: {e}")
            return {}

    def chat(
        self,
        messages: List[Dict[str, str]],
    ) -> ChatResponse:
        """Sends a chat completion request to Ollama.

        Args:
          messages: List of message objects with 'role' and 'content'.

        Returns:
          The raw response from the Ollama client.

        Raises:
          ollama.OllamaError: If the request fails.
        """
        # Prepare parameters for the call
        # Create a copy of model options to avoid modifying the original
        options = self._model_options.copy()

        params: Dict[str, Any] = {
            "model": self._model_name,
            "options": options,
            "stream": self._stream,
            "think": self._think,
        }

        # If 'think' is not set, use the output schema
        # For some reason Ollama does not accept 'think' and 'format' together
        if not self._think:
            params["format"] = get_output_schema()

        try:
            # Debug: Show parameters being sent
            print(f"Sending parameters: {params}")

            # Perform the chat completion
            response = self._client.chat(
                messages=messages,
                **params,
            )
            return response

        except ResponseError as e:
            # Check if the error is due to model not found
            if e.status_code == 404 and "not found" in str(e):
                print(
                    f"Model '{self._model_name}' not found. "
                    "Attempting to pull..."
                )
                self._pull_model(self._model_name)

                # Retry the chat completion after pulling the model
                print(
                    f"Retrying chat completion with model: {self._model_name}"
                )
                response = self._client.chat(
                    messages=messages,
                    **params,
                )
                return response
            else:
                # Re-raise other ResponseErrors
                raise

    def parse_label(self, result: ChatResponse) -> Label:
        """Parses the response to extract the label.

        Args:
          result: The response from the chat completion.

        Returns:
          A Label object containing the parsed label.
        """
        pred_label = None
        try:
            content = (
                result.get("message", {})
                .get("content", "{}")
                .strip()
                .replace("\n", "")
                .replace("\t", "")
            )
            if len(content) == 0:
                return pred_label
            pred_label = Label.model_validate_json(content)
        except Exception as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Raw response: {content}")
            if content.startswith("{{"):
                content = content.replace("{{", "{").replace("}}", "}")
            try:
                pred_label = Label.model_validate_json(content)
            except Exception as e:
                print(f"Error decoding JSON response: {e}")
                print(f"Raw response: {content}")
        return pred_label

    def get_ollama_messages(
        self, claim: str, evidence: str, prompt_type: PromptType
    ) -> List[Dict[str, Any]]:
        """Generates messages for the Ollama chat based on claim and evidence.

        Args:
            claim: The claim to be fact-checked.
            evidence: The evidence related to the claim.

        Returns:
            A list of dictionaries representing the chat messages.
        """
        return [
            {
                "role": "system",
                "content": get_fact_check_system_prompt(),
            },
            {
                "role": "user",
                "content": get_fact_check_user_prompt(
                    claim,
                    evidence,
                    prompt_type=prompt_type,
                ),
            },
        ]


def process_claim(
    llm, claim, evidence, perturb_type, shot, perturb
):  # noqa: C901
    """Process a single claim with the LLM and return results.

    Args:
        llm: The OllamaLLM instance.
        claim: The claim text to evaluate.
        evidence: The evidence text.
        pertrub_type: The perturbation type.
        shot: Shot type to use ("0-shot", "2-shot", "neg-shot").

    Returns:
        Dictionary with response and prediction results.
    """
    if not claim or not evidence:
        print(f"Claim or evidence is missing for {perturb_type}.")
        return {}

    # Determine message creation based on shot type
    if shot == "2-shot":
        messages = llm.get_ollama_messages(claim, evidence, PromptType.FEW_SHOT)
    elif shot == "0-shot":
        messages = llm.get_ollama_messages(
            claim, evidence, PromptType.ZERO_SHOT
        )
    elif shot == "neg-shot":
        messages = llm.get_ollama_messages(
            claim, evidence, PromptType.NEGATIVE_SYNTHETIC_ALL
        )
    elif shot == "neg-same-type-shot":
        if perturb == "normalized_claim":
            messages = llm.get_ollama_messages(
                claim, evidence, PromptType.FEW_SHOT
            )
        else:
            try:
                prompt_type = PromptType(perturb)
            except KeyError:
                print(
                    f"Invalid perturbation type '{perturb}'. Falling back to NEGATIVE_SYNTHETIC_ALL."  # noqa E501
                )

            try:
                messages = llm.get_ollama_messages(claim, evidence, prompt_type)
            except Exception as e:
                print(
                    f"Prompt type {perturb} not found. "
                    "Falling back to NEGATIVE_SYNTHETIC_ALL."
                )
                print(e)

    # Get model response and process results
    result = llm.chat(messages)
    response_data = result.model_dump()

    # Parse the prediction label
    pred_label = llm.parse_label(result)
    pred_label_value = (
        pred_label.model_dump().get("label", None) if pred_label else None
    )

    return {
        f"{perturb_type}_ollama_response": response_data,
        f"{perturb_type}_pred_label": pred_label_value,
    }


def main() -> None:  # noqa: C901
    """Example usage of OllamaLLM.

    Loads configuration from 'config.yaml' and runs a sample chat.

    Returns:
      None.
    """
    default_perturbations = {
        "approximation",
        "negative_number",
        "numeration",
        "random_replacement",
        "range",
        "masking",
    }
    default_models = {"deepseek_r1", "qwen3_32b", "llama3.3_70b"}
    parser = argparse.ArgumentParser(description="Run OllamaLLM experiments.")

    parser.add_argument(
        "--labels",
        nargs="+",
        choices=["true", "false"],
        default=["false"],
        help="Labels to use for the experiment (e.g., 'true', 'false', or both).",  # noqa E501
    )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Define the configs directory
    configs_dir = "code/llm_eval/ollama_configs"

    # Get all YAML files in the configs directory
    # Add command-line arguments for models and perturbations
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of model names to run experiments for",
        default=list(default_models),
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        help="List of perturbation types to run experiments for",
        default=default_perturbations,
    )
    parser.add_argument("--host", type=str, default="http://localhost:11434")

    parser.add_argument(
        "--shot",
        type=str,
        choices=["0-shot", "2-shot", "neg-shot", "neg-same-type-shot"],
        default="2-shot",
        help="Shot type to use for the experiment.",
    )
    args = parser.parse_args()
    host = args.host
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Define the configs directory
    configs_dir = "code/llm_eval/ollama_configs"

    # Get all YAML files in the configs directory that match criteria
    config_files = []
    for model in args.models:
        config_file = f"{configs_dir}/{model}.yaml"
        config_files.append(config_file)

    print(config_files)
    if not config_files:
        print(f"No matching configuration files found in {configs_dir}")
        return

    # Define default perturbations if none specified

    # Use specified perturbations or default set
    perturbations = (
        args.perturbations if args.perturbations else default_perturbations
    )
    shot = args.shot
    for config_path in config_files:
        print(f"==Experiment with config: {os.path.basename(config_path)}==")

        llm = OllamaLLM(config_path=config_path, host=host)

        for perturb in perturbations:
            for label in args.labels:
                json_file = f"data/raw_data_jsons/{perturb}_{label}.json"
                print(
                    f"\n===== Processing file: {os.path.basename(json_file)} ====="  # noqa E501
                )
                model_name = (
                    llm._model_name + "_think"
                    if llm._think
                    else llm._model_name
                )
                out_file_path = f"results/ollama_runs/{shot}/{perturb}_{label}_{model_name}_results.jsonl"  # noqa E501

                with open(out_file_path, "w") as out_file:
                    with open(json_file) as f:
                        data = json.load(f)
                        for item in data:
                            claims = {}
                            # It's enough to have one normalized_claim
                            # for one perturbation
                            if perturb == "numeration":
                                claims["normalized_claim"] = item.get(
                                    "normalized_claim"
                                )
                            perturb_false_claim = perturb + "_false"
                            perturb_true_claim = perturb + "_true"
                            if perturb_false_claim in item:
                                claims[perturb_false_claim] = item[
                                    perturb_false_claim
                                ]
                            if perturb_true_claim in item:
                                claims[perturb_true_claim] = item[
                                    perturb_true_claim
                                ]
                            print(claims)

                            # Process claims in parallel
                            evidence = item.get("doc", "").strip()
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=4
                            ) as executor:
                                future_to_pertrub = {
                                    executor.submit(
                                        process_claim,
                                        llm,
                                        claim,
                                        evidence,
                                        pertrub_type,
                                        shot,
                                        perturb,
                                    ): pertrub_type
                                    for pertrub_type, claim in claims.items()
                                    if claim and evidence
                                }

                                for future in concurrent.futures.as_completed(
                                    future_to_pertrub
                                ):
                                    pertrub_type = future_to_pertrub[future]
                                    try:
                                        results = future.result()
                                        item.update(results)
                                    except Exception as e:
                                        print(
                                            f"Processing {pertrub_type} generated an exception: {e}"  # noqa E501
                                        )
                            print(item)
                            out_file.write(json.dumps(item) + "\n")
                            out_file.flush()


if __name__ == "__main__":
    main()

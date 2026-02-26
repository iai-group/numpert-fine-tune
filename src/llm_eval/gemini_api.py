#!/usr/bin/env python3
"""Script to process OpenAI JSONL format input using Google Gemini API."""

import argparse
import json
import sys
import time
import os
from typing import Dict, List, Optional, Any, Set
import google.genai as genai  # type: ignore[import]
from google.genai import types  # type: ignore[import]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate claims from JSONL input using Google Gemini with reasoning."  # noqa: E501
    )
    parser.add_argument(
        "-i", "--input_file", help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output_file", help="Path to the output JSONL file"
    )
    parser.add_argument("--api_key", help="Google API key")
    parser.add_argument(
        "--num_elements", type=int, help="Number of elements to evaluate"
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-flash",
        help="Gemini model name to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for model generation (0.0 to 2.0, default: 0",
    )
    parser.add_argument(
        "--show_model_output",
        action="store_true",
        help="Print the model's input messages",
    )
    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Enable reasoning mode with thinking",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=8192,
        help="Thinking budget (0=no thinking, 1=dynamic)",
    )
    parser.add_argument(
        "--show_thoughts",
        action="store_true",
        help="Include model thoughts in output and display",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed records)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save progress every N records",
    )
    return parser.parse_args()


def read_jsonl(
    file_path: str, num_elements: Optional[int] = None
) -> List[Dict]:
    """Read JSONL file and return list of dictionaries.

    Args:
        file_path: Path to the JSONL file.
        num_elements: Maximum number of elements to read (optional).

    Returns:
        A list of dictionaries representing the JSONL data.
    """
    data = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if num_elements and i >= num_elements:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                continue
    return data


def get_processed_ids(output_file: str) -> Set[str]:
    """Get set of custom_ids that have already been processed.

    Args:
        output_file: Path to the output JSONL file.

    Returns:
        A set of custom_ids that have already been processed.
    """
    processed_ids: set[str] = set()

    if not os.path.exists(output_file):
        return processed_ids

    try:
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if "custom_id" in record:
                        processed_ids.add(record["custom_id"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return processed_ids


def save_checkpoint(
    output_file: str, results: List[Dict], append_mode: bool = False
) -> None:
    """Save results to output file.

    Args:
        output_file: Path to the output JSONL file.
        results: List of results to save.
        append_mode: If True, appends to the file; otherwise, overwrites it.
    """
    mode = "a" if append_mode else "w"
    with open(output_file, mode) as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI message format to a single prompt for Gemini.

    Args:
        messages: List of message dictionaries.

    Returns:
        A single string prompt for Gemini.
    """
    prompt_parts = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    return "\n\n".join(prompt_parts)


def process_with_gemini(  # noqa: C901
    client: genai.Client,
    prompt: str,
    model_name: str,
    temperature: float,
    use_reasoning: bool,
    thinking_budget: int,
    show_thoughts: bool,
) -> Dict[str, Any]:  # noqa: C901
    """Process prompt with Gemini and return structured response.

    Args:
        client: Gemini client instance.
        prompt: The prompt to process.
        model_name: The name of the Gemini model to use.
        temperature: Temperature for model generation.
        use_reasoning: Whether to enable reasoning mode.
        thinking_budget: Budget for reasoning tokens.
        show_thoughts: Whether to display model thoughts.

    Returns:
        A dictionary containing the response and token usage.
    """
    thoughts = ""
    answer = ""
    thoughts_tokens = 0
    output_tokens = 0
    total_tokens = 0
    last_chunk = None

    try:
        if use_reasoning:
            # Use reasoning mode with thinking
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True, thinking_budget=thinking_budget
                ),
                temperature=temperature,
            )
        else:
            # Regular mode without thinking
            config = types.GenerateContentConfig(temperature=temperature)

        # Generate content with streaming
        for chunk in client.models.generate_content_stream(
            model=model_name, contents=prompt, config=config
        ):
            last_chunk = chunk  # Keep reference to last chunk

            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    elif hasattr(part, "thought") and part.thought:
                        if show_thoughts and not thoughts:
                            print("\n[Thoughts:]", end=" ")
                        if show_thoughts:
                            print(part.text, end="")
                        thoughts += part.text
                    else:
                        if not answer:
                            if show_thoughts:
                                print("\n[Answer:]", end=" ")
                        if show_thoughts:
                            print(part.text, end="")
                        answer += part.text

        # Get token counts from the last chunk
        if (
            last_chunk
            and hasattr(last_chunk, "usage_metadata")
            and last_chunk.usage_metadata
        ):
            # Get thoughts token count
            if hasattr(last_chunk.usage_metadata, "thoughts_token_count"):
                thoughts_tokens = (
                    last_chunk.usage_metadata.thoughts_token_count or 0
                )

            # Get total token count (includes both thoughts and output)
            if hasattr(last_chunk.usage_metadata, "total_token_count"):
                total_tokens = last_chunk.usage_metadata.total_token_count or 0
            elif hasattr(last_chunk.usage_metadata, "candidates_token_count"):
                # If total_token_count not available, use candidates_token_count
                total_tokens = (
                    last_chunk.usage_metadata.candidates_token_count or 0
                )

            # Calculate output tokens (excluding thoughts)
            output_tokens = max(0, total_tokens - thoughts_tokens)

            # Debug token counts if verbose
            if show_thoughts:
                print(
                    f"\n[Debug - Raw token counts: total={total_tokens}, thoughts={thoughts_tokens}, calculated_output={output_tokens}]"  # noqa: E501
                )

        return {
            "answer": answer.strip(),
            "thoughts": thoughts,
            "thoughts_tokens": thoughts_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error": None,
        }

    except Exception as e:
        return {
            "answer": "",
            "thoughts": "",
            "thoughts_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
        }


def create_output_record(
    custom_id: str, result: Dict[str, Any], model_name: str
) -> Dict[str, Any]:
    """Create output record in OpenAI batch format.

    Args:
        custom_id: The custom identifier for the record.
        result: The result dictionary from Gemini processing.
        model_name: The name of the Gemini model used.

    Returns:
        A dictionary formatted for JSONL output.
    """
    if result["error"]:
        return {
            "custom_id": custom_id,
            "response": None,
            "error": {
                "message": result["error"],
                "type": "api_error",
                "code": None,
            },
        }

    # Build usage dict
    usage = {
        "completion_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"],
    }

    # Add reasoning tokens if available
    if result["thoughts_tokens"] > 0:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": result["thoughts_tokens"],
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }

    response_body: Dict[str, Any] = {
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["answer"],
                    "refusal": None,
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "service_tier": "default",
        "system_fingerprint": None,
    }

    # Optionally include thoughts in the response
    if result["thoughts"] and result["thoughts"].strip():
        response_body["choices"][0]["message"][
            "thoughts"
        ] = result[  # noqa: E501
            "thoughts"
        ]

    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": response_body},
        "error": None,
    }


def main() -> None:  # noqa: C901
    """Main function to process JSONL input using Google Gemini."""
    args = parse_arguments()

    # Validate temperature value
    if args.temperature < 0.0 or args.temperature > 2.0:
        print(
            f"Warning: Temperature {args.temperature} is outside recommended range (0.0-2.0)"  # noqa: E501
        )

    # Initialize Gemini client
    if not args.api_key:
        print(
            "Error: Google API key is required. Use --api_key or set GOOGLE_API_KEY environment variable."  # noqa: E501
        )
        sys.exit(1)

    client = genai.Client(api_key=args.api_key)

    # Read input file
    print(f"Reading input file: {args.input_file}")
    input_data = read_jsonl(args.input_file, args.num_elements)
    print(f"Loaded {len(input_data)} records")

    # Check for resume functionality
    processed_ids = set()
    append_mode = False

    if args.resume:
        processed_ids = get_processed_ids(args.output_file)
        if processed_ids:
            print(
                f"Resume mode: Found {len(processed_ids)} already processed records"  # noqa: E501
            )
            append_mode = True
        else:
            print("Resume mode: No existing output file found, starting fresh")

    # Filter out already processed records
    remaining_records = []
    skipped_count = 0

    for record in input_data:
        custom_id = record.get(
            "custom_id", f"record_{input_data.index(record)}"
        )
        if custom_id in processed_ids:
            skipped_count += 1
        else:
            remaining_records.append(record)

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed records")

    if not remaining_records:
        print("All records have already been processed!")
        return

    print(f"Processing {len(remaining_records)} remaining records")
    print(f"Using temperature: {args.temperature}")

    # Process each remaining record
    results_batch = []
    processed_count = 0

    try:
        for i, record in enumerate(remaining_records):
            custom_id = record.get(
                "custom_id",
                f"record_{len(input_data) - len(remaining_records) + i}",
            )
            current_position = len(input_data) - len(remaining_records) + i + 1

            print(
                f"\nProcessing {current_position}/{len(input_data)}: {custom_id}"  # noqa: E501
            )

            # Extract messages from the body
            body = record.get("body", {})
            messages = body.get("messages", [])

            if args.show_model_output:
                print("Messages:", json.dumps(messages, indent=2))

            # Convert messages to prompt
            prompt = convert_messages_to_prompt(messages)

            if args.show_model_output:
                print("Converted prompt:", prompt)

            # Process with Gemini
            result = process_with_gemini(
                client=client,
                prompt=prompt,
                model_name=args.model_name,
                temperature=args.temperature,
                use_reasoning=args.use_reasoning,
                thinking_budget=args.thinking_budget,
                show_thoughts=args.show_thoughts,
            )

            # Create output record
            output_record = create_output_record(
                custom_id, result, args.model_name
            )
            results_batch.append(output_record)
            processed_count += 1

            if args.show_thoughts and result["thoughts_tokens"] > 0:
                print(
                    f"\n[Tokens - Thoughts: {result['thoughts_tokens']}, Output: {result['output_tokens']}]"  # noqa: E501
                )

            # Save checkpoint periodically
            if processed_count % args.checkpoint_interval == 0:
                print(
                    f"\n[Checkpoint] Saving progress after {processed_count} records..."  # noqa: E501
                )
                save_checkpoint(args.output_file, results_batch, append_mode)
                results_batch = []  # Clear batch after saving
                append_mode = True  # Switch to append mode after first save

            # Add a small delay to avoid rate limiting
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(
            f"\n\nInterrupted! Saving progress after {processed_count} records..."  # noqa: E501
        )
        if results_batch:
            save_checkpoint(args.output_file, results_batch, append_mode)
        print("Progress saved. You can resume later using --resume flag.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print(f"Saving progress after {processed_count} records...")
        if results_batch:
            save_checkpoint(args.output_file, results_batch, append_mode)
        print("Progress saved. You can resume later using --resume flag.")
        raise

    # Save any remaining results
    if results_batch:
        print(f"\nSaving final batch of {len(results_batch)} records...")
        save_checkpoint(args.output_file, results_batch, append_mode)

    total_processed = len(processed_ids) + processed_count
    print(
        f"\nSuccessfully processed {total_processed} total records ({processed_count} in this run)"  # noqa: E501
    )


if __name__ == "__main__":
    main()

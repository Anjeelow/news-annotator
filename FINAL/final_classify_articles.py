import os
import json
import time
import argparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Defaults
DEFAULT_INPUT_JSON = os.getenv("INPUT_JSON", "FINAL/new_kept_articles.json")
DEFAULT_RULES_FILE = os.getenv("RULES_FILE", "annotation_rules_new.txt")
DEFAULT_OUTPUT_JSON = os.getenv("OUTPUT_JSON", "FINAL/gpt5p2_mini_labeled_new_kept_results.json")
DEFAULT_PROGRESS = os.getenv("PROGRESS", "FINAL/annotation_progress.txt")


try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


CATEGORIES = [
    "Biased For the Government",
    "Slightly Biased For the Government",
    "Neutral",
    "Slightly Biased Against the Government",
    "Biased Against the Government"
]


# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------

def load_rules(rules_filepath: str) -> str:
    try:
        with open(rules_filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Rules file '{rules_filepath}' not found!")
        exit(1)


def load_articles_from_json(json_filepath: str) -> list:
    """Load flat-array JSON: [ {article}, {article}, ... ]"""
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Articles file '{json_filepath}' not found!")
        exit(1)

    if not isinstance(data, list):
        print("ERROR: Input JSON must be a flat list of article objects.")
        exit(1)

    # Ensure articles have at least minimal fields
    for article in data:
        if "publisher" not in article:
            article["publisher"] = "unknown"

    return data


def create_classification_prompt(article_text: str) -> str:
    return f"""You are an expert news article bias classifier for Philippine government-related news.

Article body:
{article_text}
"""


def _parse_model_output(text: str) -> Dict[str, str]:
    VALID_3_POINT = {"BA", "N", "BF"}
    VALID_5_POINT = {"BA", "SBA", "N", "SBF", "BF"}

    raw_lines = (text or "").splitlines()
    lines = [ln.strip() for ln in raw_lines]

    # Ensure exactly four lines
    while len(lines) < 4:
        lines.append("")

    # Enforce format
    if lines[0] not in VALID_3_POINT:
        raise ValueError(f"Invalid 3-point label: {lines[0]}")

    if lines[1] not in VALID_5_POINT:
        raise ValueError(f"Invalid 5-point label: {lines[1]}")

    return {
        "label": "SUCCESS",
        "3_point_label": lines[0],              # BA | N | BF
        "5_point_label": lines[1],              # BA | SBA | N | SBF | BF
        "biased_for_indicators": lines[2],      # quoted phrases or empty
        "biased_against_indicators": lines[3],  # quoted phrases or empty
    }



def _get_client() -> Optional[Any]:
    if OpenAI is None:
        print("ERROR: Missing openai package. Install with:\n  pip install openai")
        return None
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return None

    try:
        return OpenAI()
    except Exception as e:
        print(f"ERROR initializing OpenAI client: {e}")
        return None


def classify_article(
    article_text: str,
    rules: str,
    model: str = "gpt-5-mini"
) -> Optional[Dict[str, str]]:
    """Call OpenAI API and return parsed classification fields plus raw model text."""

    client = _get_client()
    if client is None:
        return None

    user_prompt = create_classification_prompt(article_text)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert news article bias classifier for Philippine government-related news.\n\n"
                "You must follow the rules below with highest priority and must not deviate from the specified output format.\n\n"
                "If your output violates the required format, you must correct it before responding.\n\n"
                f"{rules}"
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    for attempt in range(3):
        raw_text = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=32768,
            )

            # ------------------------------------------------------------------
            # 🛡 BULLETPROOF RESPONSE EXTRACTION
            # ------------------------------------------------------------------
            if getattr(resp, "choices", None):
                choice = resp.choices[0]

                if hasattr(choice, "message") and choice.message:
                    c = choice.message.content
                    if isinstance(c, str):
                        raw_text = c
                    elif isinstance(c, list):
                        raw_text = "".join(
                            part.get("text", "")
                            for part in c
                            if isinstance(part, dict)
                        )

                if not raw_text and hasattr(choice, "content"):
                    c = choice.content
                    if isinstance(c, str):
                        raw_text = c
                    elif isinstance(c, list):
                        raw_text = "".join(
                            part.get("text", "")
                            for part in c
                            if isinstance(part, dict)
                        )

                if not raw_text and hasattr(choice, "text"):
                    raw_text = choice.text or ""

                if not raw_text and isinstance(choice, dict):
                    raw_text = (
                        choice.get("message", {}).get("content", "") or
                        choice.get("content", "") or
                        choice.get("text", "")
                    )

            raw_text = (raw_text or "").strip()
            # ------------------------------------------------------------------

            if not raw_text:
                raise ValueError("Empty model response")

            parsed = _parse_model_output(raw_text)
            parsed["raw"] = raw_text
            return parsed

        except Exception as e:
            if attempt == 2:
                if raw_text:
                    parsed = _parse_model_output(raw_text)
                    parsed["raw"] = raw_text
                    return parsed

                print(f"OpenAI API error (attempt {attempt+1}/3): {e}. No model output; aborting this item.")
                exit(1)

            wait = 1.5 * (attempt + 1)
            print(f"OpenAI API error (attempt {attempt+1}/3): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return None




# -------------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------------

def main(json_filepath: str, rules_filepath: str, output_filepath: str, model_name: str):
    """Run the end-to-end classification pipeline.

    Args:
        json_filepath: Path to input JSON. Supports either:
            - Grouped dict: {"INQUIRER": [...], "GMA": [...], ...}
            - Flat list: [ {article}, {article}, ... ]
        rules_filepath: Path to the annotation rules text file.
        output_filepath: Path to write a flat JSON list of results.
        model_name: OpenAI model name (e.g., "gpt-5-mini").

    Output file shape:
        A flat list of article dicts where each result contains original fields
        plus: label, reasoning, primary_marker, confidence, raw_model_output.
        Items with very short/empty bodies are labeled "SKIPPED". API failures
        without model output are labeled "ERROR" with details.
    """
    import json
    import time

    # -------------------------------------------------------------
    # 1) Load rules
    # -------------------------------------------------------------
    print(f"Loading rules from {rules_filepath}...")
    rules = load_rules(rules_filepath)

    # -------------------------------------------------------------
    # 2) Load and normalize input articles
    #    - Accept grouped-by-publisher or flat list
    #    - Ensure `publisher` is set
    #    - Drop `replacementvalue`
    # -------------------------------------------------------------
    existing_results = []
    processed_urls = set()
    processed_count = 0

    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

            for item in existing_results:
                url = item.get("article_url")
                if url:
                    processed_urls.add(url)

            processed_count = len(existing_results)
            print(f"Resuming: {processed_count} articles already processed.")
        except:
            print("Warning: Failed to load existing output; starting fresh.")

    print(f"Loading articles from {json_filepath}...")
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Articles file '{json_filepath}' not found!")
        exit(1)

    articles = []

    # Case 1: grouped by publisher e.g., {"INQUIRER": [...], "GMA": [...], ...}
    if isinstance(raw_data, dict):
        for publisher, items in raw_data.items():
            if not isinstance(items, list):
                continue
            for article in items:
                if not isinstance(article, dict):
                    continue
                # Ensure publisher is present; prefer explicit field if given
                if "publisher" not in article or not article.get("publisher"):
                    article["publisher"] = str(publisher)
                # Drop field not needed for classification if present
                article.pop("replacementvalue", None)
                articles.append(article)
    # Case 2: legacy flat list [ {article}, ... ]
    elif isinstance(raw_data, list):
        for article in raw_data:
            if not isinstance(article, dict):
                continue
            if "publisher" not in article:
                article["publisher"] = "unknown"
            article.pop("replacementvalue", None)
            articles.append(article)
    else:
        print("ERROR: Input JSON must be either a flat list or a mapping of publisher -> list of article objects.")
        exit(1)

    print(f"Loaded {len(articles)} articles.\n")

    # -------------------------------------------------------------
    # 3) Classify articles
    # -------------------------------------------------------------
    results = []
    skipped = 0
    errors = 0

    for idx, article in enumerate(articles, 1):
        url = article.get("article_url")
        # Track last processed URL
        with open("annotation_progress.txt", "w", encoding="utf-8") as f:
            f.write(url or f"INDEX:{idx}")

        # Skip if previously processed
        if url and url in processed_urls:
            print(f"[{idx}/{len(articles)}] Skipping (already processed)")
            continue
        body = article.get("article_body", "")
        if not body or len(body.strip()) < 50:
            print(f"[{idx}/{len(articles)}] SKIPPED (too short)")
            skipped += 1
            results.append({**article, "label": "SKIPPED"})
            continue

        print(f"[{idx}/{len(articles)}] Classifying…")

        info = classify_article(body, rules, model_name)

        # info is None => client/API fatal (no model response)
        if info is None:
            errors += 1
            results.append({
                **article,
                "label": "ERROR",
                "error": "OpenAI client or API failure (no model response)",
                "raw_model_output": ""
            })
            # Save progress after every article
            safe_save = existing_results + results
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(safe_save, f, indent=2, ensure_ascii=False)

            continue

        # If there was a model response but parser couldn't find label
        label = (info.get("label") or "").strip()
        raw_text = info.get("raw", "")

        if not label:
            errors += 1
            results.append({
                **article,
                "label": "ERROR",
                "error": "Model returned output but no parsable label. See raw_model_output.",
                "raw_model_output": raw_text,
                "parsed_fields": {
                    "3_point_label": info.get("3_point_label", ""),
                    "5_point_label": info.get("5_point_label", ""),
                    "biased_for_indicators": info.get("biased_for_indicators", ""),
                    "biased_against_indicators": info.get("biased_against_indicators", "")
                }
            })
            # Save progress after every article
            safe_save = existing_results + results
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(safe_save, f, indent=2, ensure_ascii=False)


            continue

        # Normal success path
        results.append({
            **article,
            "label": info.get("label", ""),
            "3_point_label": info.get("3_point_label", ""),
            "5_point_label": info.get("5_point_label", ""),
            "biased_for_indicators": info.get("biased_for_indicators", ""),
            "biased_against_indicators": info.get("biased_against_indicators", ""),
            "raw_model_output": raw_text
        })

        # Save progress after every article
        safe_save = existing_results + results
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(safe_save, f, indent=2, ensure_ascii=False)


    # -------------------------------------------------------------
    # 4) Save results + print summary
    # -------------------------------------------------------------
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===============================")
    print(f"Saved results to: {output_filepath}")
    print("===============================\n")

    # Stats
    print("SUMMARY:")
    print(f"  Total articles:   {len(articles)}")
    print(f"  Classified:       {len(articles) - skipped - errors}")
    print(f"  Skipped:          {skipped}")
    print(f"  Errors:           {errors}")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News article bias classifier.")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--rules", default=DEFAULT_RULES_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", default="gpt-5-mini")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set!")
        exit(1)

    main(args.input, args.rules, args.output, args.model)

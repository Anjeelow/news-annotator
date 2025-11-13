import os
import json
import time
import argparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Defaults
DEFAULT_INPUT_JSON = os.getenv("INPUT_JSON", "verification_sample_articles.json")
DEFAULT_RULES_FILE = os.getenv("RULES_FILE", "annotation_rules_final.txt")
DEFAULT_OUTPUT_JSON = os.getenv("OUTPUT_JSON", "gpt5_mini_labeled_verification_results.json")

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


def create_classification_prompt(article_text: str, rules: str) -> str:
    return f"""You are an expert news article bias classifier for Philippine government-related news.

{rules}

INSTRUCTIONS:
1. Read article body only (ignore headline)
2. Count subtle markers present
3. Assess attribution order and voice consistency
4. Check balance of space/detail between government and opposition
5. Select ONE category that best describes the article's bias

Output format:
Classification: [Select one of the 5 categories above]
Confidence: [High/Medium/Low]
Primary marker: [Which specific marker/pattern drove your decision]
Reasoning: [2-3 sentences explaining your classification]

Article body:
{article_text}
"""


def _normalize_label(text: str) -> str:
    if not text:
        return ""
    t = text.strip().strip('"').strip("'.:; ")

    # Exact match
    if t in CATEGORIES:
        return t

    # Case-insensitive match
    for cat in CATEGORIES:
        if t.lower() == cat.lower():
            return cat

    # Numeric mapping
    mapping = {str(i + 1): cat for i, cat in enumerate(CATEGORIES)}
    if t in mapping:
        return mapping[t]

    return t


def _parse_model_output(text: str) -> Dict[str, str]:
    label = ""
    confidence = ""
    primary = ""
    reasoning = ""

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]

    for ln in lines:
        lower = ln.lower()
        if lower.startswith("classification:") or lower.startswith("label:"):
            label = ln.split(":", 1)[1].strip()
        elif lower.startswith("confidence:"):
            confidence = ln.split(":", 1)[1].strip()
        elif lower.startswith("primary marker:"):
            primary = ln.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            reasoning = ln.split(":", 1)[1].strip()

    return {
        "label": label,
        "confidence": confidence,
        "primary_marker": primary,
        "reasoning": reasoning,
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


def classify_article(article_text: str, rules: str, model: str = "gpt-5-mini") -> Optional[Dict[str, str]]:
    """Call OpenAI API and return parsed classification fields plus raw model text.

    Returns:
      - dict with keys: label, confidence, primary_marker, reasoning, raw
      - or None if the client couldn't be created or there was an API-level failure without a model response
    """
    client = _get_client()
    if client is None:
        return None

    prompt = create_classification_prompt(article_text, rules)

    for attempt in range(3):
        raw_text = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=32768,
            )

            # ------------------------------------------------------------------
            # 🛡 BULLETPROOF RESPONSE EXTRACTION (works for all OpenAI responses)
            # ------------------------------------------------------------------
            if getattr(resp, "choices", None):
                choice = resp.choices[0]

                # 1) message.content (string or list)
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

                # 2) top-level .content (string or list; used by gpt-5-mini)
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

                # 3) legacy compatibility .text
                if not raw_text and hasattr(choice, "text"):
                    raw_text = choice.text or ""

                # 4) dict fallback (rare)
                if not raw_text and isinstance(choice, dict):
                    raw_text = (
                        choice.get("message", {}).get("content", "") or
                        choice.get("content", "") or
                        choice.get("text", "")
                    )

            raw_text = (raw_text or "").strip()
            # ------------------------------------------------------------------
            # END EXTRACTION BLOCK
            # ------------------------------------------------------------------

            if not raw_text:
                raise ValueError("Empty model response")

            parsed = _parse_model_output(raw_text)
            parsed["label"] = _normalize_label(parsed.get("label", ""))
            parsed["raw"] = raw_text
            return parsed

        except Exception as e:
            if attempt == 2:
                if raw_text:
                    parsed = _parse_model_output(raw_text)
                    parsed["label"] = _normalize_label(parsed.get("label", ""))
                    parsed["raw"] = raw_text
                    return parsed

                print(f"OpenAI API error (attempt {attempt+1}/3): {e}. No model output; aborting this item.")
                return None

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
                    "confidence": info.get("confidence", ""),
                    "primary_marker": info.get("primary_marker", ""),
                    "reasoning": info.get("reasoning", "")
                }
            })
            continue

        # Normal success path
        results.append({
            **article,
            "label": label,
            "reasoning": info.get("reasoning", ""),
            "primary_marker": info.get("primary_marker", ""),
            "confidence": info.get("confidence", ""),
            "raw_model_output": raw_text
        })

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

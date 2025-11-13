import os
import json
import time
from dotenv import load_dotenv
import argparse
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Defaults from environment so you can "Run" without CLI args
DEFAULT_INPUT_JSON = os.getenv("INPUT_JSON", "verification_sample_articles.json")
DEFAULT_RULES_FILE = os.getenv("RULES_FILE", "annotation_rules_final.txt")
DEFAULT_OUTPUT_JSON = os.getenv("OUTPUT_JSON", "gpt5_mini_labeled_verification_results.json")
_sample_env = os.getenv("SAMPLE_SIZE")
try:
    DEFAULT_SAMPLE_SIZE = int(_sample_env) if _sample_env else None
except ValueError:
    DEFAULT_SAMPLE_SIZE = None
CONSUME_INPUT_DEFAULT = (os.getenv("CONSUME_INPUT", "false").strip().lower() in {"1", "true", "yes"})
# OpenAI SDK (expects OPENAI_API_KEY in environment)
try:
    from openai import OpenAI
except Exception:
    # Keep import-time errors from crashing static tooling; runtime will surface a clear message
    OpenAI = None  # type: ignore

CATEGORIES = [
    "Biased For the Government",
    "Slightly Biased For the Government",
    "Neutral",
    "Slightly Biased Against the Government",
    "Biased Against the Government"
]

def load_rules(rules_filepath: str) -> str:
    """Load annotation rules from text file."""
    try:
        with open(rules_filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Rules file '{rules_filepath}' not found!")
        exit(1)

def create_classification_prompt(article_text: str, rules: str) -> str:
    """Create the classification prompt for the model (requests structured output)."""
    prompt = f"""You are an expert news article bias classifier for Philippine government-related news.

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
Statements and words that indicate bias against: [Phrases and words that indicate bias against the government, negative statements about the government or sentences leaning against government views.] 
Statements and words that indicate bias for: [Phrases and words that indicate bias for the government, positive statements about the government or sentences leaning towards government views.] 

Article body:
{article_text}
"""
    return prompt

def _normalize_label(text: str) -> str:
    """Normalize model output to one of the allowed categories when possible."""
    if not text:
        return ""
    t = text.strip().strip('"').strip("'.:; ")
    # Exact match first
    if t in CATEGORIES:
        return t
    # Case-insensitive match
    for cat in CATEGORIES:
        if t.lower() == cat.lower():
            return cat
    # Handle numeric answers "1".."5"
    mapping: Dict[str, str] = {
        "1": CATEGORIES[0],
        "2": CATEGORIES[1],
        "3": CATEGORIES[2],
        "4": CATEGORIES[3],
        "5": CATEGORIES[4],
    }
    # Accept common shortened forms (without "the Government")
    simplified: Dict[str, str] = {
        "biased for": CATEGORIES[0],
        "slightly biased for": CATEGORIES[1],
        "neutral": CATEGORIES[2],
        "slightly biased against": CATEGORIES[3],
        "biased against": CATEGORIES[4],
    }
    if t in mapping:
        return mapping[t]
    for k, v in simplified.items():
        if t.lower() == k:
            return v
    # Try to map substrings
    t_low = t.lower()
    for k, v in simplified.items():
        if k in t_low:
            return v
    for cat in CATEGORIES:
        if cat.lower() in t_low:
            return cat
    return t


def _get_client() -> Optional[Any]:
    """Create an OpenAI client from env var. Returns None and prints a helpful message if not available."""
    if OpenAI is None:
        print("ERROR: The 'openai' Python package is not installed. Install it with: pip install openai")
        return None
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set! Set it in PowerShell with:\n  $env:OPENAI_API_KEY=\"your_key_here\"")
        return None
    try:
        return OpenAI()
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return None


def _parse_model_output(text: str) -> Dict[str, str]:
    """Extract Label, Primary marker, and Reasoning from model output.
    Accepts minor variations like 'Classification:' for label and case differences.
    """
    label = ""
    confidence = ""
    primary = ""
    reasoning = ""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("label:") or lower.startswith("classification:"):
            label = ln.split(":", 1)[1].strip()
        elif lower.startswith("confidence:"):
            confidence = ln.split(":", 1)[1].strip()
        elif lower.startswith("primary marker:") or lower.startswith("primary-marker:"):
            primary = ln.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            reasoning = ln.split(":", 1)[1].strip()
    # If reasoning not captured on a single line, try to join remaining text after 'Reasoning:'
    if not reasoning and any(ln.lower().startswith("reasoning:") for ln in lines):
        start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("reasoning:"))
        reasoning = " ".join(lines[start:])[len("Reasoning:"):].strip()
    return {"label": label, "confidence": confidence, "primary_marker": primary, "reasoning": reasoning}


def classify_article(article_text: str, rules: str, model: str = "gpt-5-mini") -> Optional[Dict[str, str]]:
    """Classify a single article using the OpenAI API and return structured info."""
    client = _get_client()
    if client is None:
        return None

    prompt = create_classification_prompt(article_text, rules)

    # Basic retry for transient failures/rate limits
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=256,
            )
            raw = resp.choices[0].message.content if resp.choices else None
            if not raw:
                print("Warning: Empty response from model")
                return None
            parsed = _parse_model_output(raw)
            label_norm = _normalize_label(parsed.get("label", ""))
            parsed["label"] = label_norm
            # Validate the response
            if not label_norm:
                print("Warning: Could not parse a label from model output")
                return None
            return parsed
        except Exception as e:
            wait = 1.5 * (attempt + 1)
            print(f"OpenAI API error (attempt {attempt+1}/3): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None

def load_articles_from_json(json_filepath: str) -> list:
    """Load articles from the JSON file."""
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            articles_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Articles file '{json_filepath}' not found!")
        exit(1)
    
    # Flatten the nested structure (source -> list of articles)
    articles = []
    for source, article_list in articles_dict.items():
        for article in article_list:
            articles.append(article)
    
    return articles

def _select_sample_diverse_by_publisher(articles: list, n: Optional[int]) -> list:
    """Select up to n articles, preferring at most 1 per publisher in the first pass.
    - If n is None, returns all articles.
    - If n < number of publishers, returns first n publishers' first articles.
    - If n > number of publishers, after taking one per publisher, continue round-robin until n.
    """
    if not n or n <= 0:
        return articles
    # Group by publisher preserving original order
    by_pub = {}
    for art in articles:
        pub = (art.get("publisher") or "unknown").strip() or "unknown"
        by_pub.setdefault(pub, []).append(art)
    pubs = list(by_pub.keys())
    out = []
    idx = 0
    while len(out) < n and pubs:
        pub = pubs[idx % len(pubs)]
        bucket = by_pub[pub]
        if bucket:
            out.append(bucket.pop(0))
        # If a bucket is exhausted, remove pub from rotation
        if not bucket:
            pubs.remove(pub)
            # Adjust idx because list shrank
            if pubs:
                idx = idx % len(pubs)
            continue
        idx += 1
    return out


def _load_existing_labeled_urls(output_filepath: str) -> set[str]:
    """Load already-labeled article URLs from an existing results file to avoid duplicates."""
    urls: set[str] = set()
    if not os.path.exists(output_filepath):
        return urls
    try:
        with open(output_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for pub, items in data.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    lbl = (it.get("label") or "").strip().upper()
                    url = (it.get("article_url") or "").strip()
                    if not url:
                        continue
                    if lbl and lbl not in {"ERROR", "SKIPPED"}:
                        urls.add(url)
    except Exception:
        # If prior file is malformed, just return empty set
        return set()
    return urls


def main(json_filepath: str, rules_filepath: str, output_filepath: str, sample_size: Optional[int] = None, consume: bool = False):
    """Main function to classify articles."""
    
    # Load rules from file
    print(f"Loading annotation rules from {rules_filepath}...")
    rules = load_rules(rules_filepath)
    
    # Load articles
    print(f"Loading articles from {json_filepath}...")
    articles = load_articles_from_json(json_filepath)
    
    # Optional: limit to sample size for testing (diverse by publisher)
    if sample_size:
        articles = _select_sample_diverse_by_publisher(articles, sample_size)
        print(f"Using sample of {len(articles)} articles (diverse by publisher, target={sample_size})")
    
    print(f"Loaded {len(articles)} articles.\n")
    
    results = []
    skipped_count = 0
    error_count = 0
    duplicate_skipped = 0

    # Build a set of already labeled URLs to avoid re-labeling/duplicates
    already_labeled_urls = _load_existing_labeled_urls(output_filepath)
    
    for idx, article in enumerate(articles, 1):
        article_url = article.get("article_url", "")
        author_name = article.get("author_name", "")
        article_headline = article.get("article_headline", "")
        date_published = article.get("date_published", "")
        article_body = article.get("article_body", "")
        publisher = article.get("publisher", "")
        
        # Skip articles without body text
        if not article_body or len(article_body.strip()) < 50:
            print(f"[{idx}/{len(articles)}] SKIPPED (insufficient text): {article_headline[:60]}...")
            results.append({
                "article_url": article_url,
                "author_name": author_name,
                "article_headline": article_headline,
                "date_published": date_published,
                "article_body": article_body,
                "publisher": publisher,
                "label": "SKIPPED"
            })
            skipped_count += 1
            continue
        
        # Skip if already labeled in previous runs
        if article_url and article_url in already_labeled_urls:
            print(f"[{idx}/{len(articles)}] SKIPPED (already labeled): {article_headline[:60]}...")
            skipped_count += 1
            duplicate_skipped += 1
            continue

        print(f"[{idx}/{len(articles)}] Classifying: {publisher} - {article_headline[:60]}...")

        info = classify_article(article_body, rules)
        
        if info is None:
            error_count += 1
        label = info.get("label") if info else None
        reasoning = info.get("reasoning") if info else None
        primary_marker = info.get("primary_marker") if info else None
        confidence = info.get("confidence") if info else None

        results.append({
            "article_url": article_url,
            "author_name": author_name,
            "article_headline": article_headline,
            "date_published": date_published,
            "article_body": article_body,
            "publisher": publisher,
            "label": label if label else "ERROR",
            "reasoning": reasoning if reasoning else "",
            "primary_marker": primary_marker if primary_marker else "",
            "confidence": confidence if confidence else ""
        })
    
    # Save results in same format as input JSON (nested by publisher)
    # Append if file exists; otherwise create new.
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as in_f:
                output_dict = json.load(in_f)
        except Exception:
            output_dict = {}
    else:
        output_dict = {}

    for result in results:
        publisher = result.get("publisher", "unknown")
        if publisher not in output_dict or not isinstance(output_dict[publisher], list):
            output_dict[publisher] = []
        output_dict[publisher].append(result)

    # Optional deduplication by article_url (keep most recent entry)
    for pub, items in list(output_dict.items()):
        if not isinstance(items, list):
            continue
        seen = {}
        # Keep last occurrence (new results were appended last)
        for i, it in enumerate(items):
            key = (it or {}).get("article_url") or f"__idx_{i}__"
            seen[key] = it
        output_dict[pub] = list(seen.values())

    with open(output_filepath, "w", encoding="utf-8") as out_f:
        json.dump(output_dict, out_f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Classification complete! Results saved to {output_filepath}")
    print(f"{'='*80}\n")
    
    # Print summary statistics
    label_counts = {}
    for result in results:
        label = result.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("CLASSIFICATION SUMMARY:")
    print("-" * 80)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print(f"\nProcessing Statistics:")
    print(f"  Total articles: {len(articles)}")
    print(f"  Successfully classified: {len(articles) - skipped_count - error_count}")
    print(f"  Skipped (insufficient text): {skipped_count}")
    print(f"  Errors: {error_count}\n")
    if duplicate_skipped:
        print(f"  Skipped (already labeled): {duplicate_skipped}")
    
    # Print by publisher
    print("CLASSIFICATION BY PUBLISHER:")
    print("-" * 80)
    publisher_stats = {}
    for result in results:
        pub = result.get("publisher", "unknown")
        label = result.get("label", "UNKNOWN")
        if pub not in publisher_stats:
            publisher_stats[pub] = {}
        publisher_stats[pub][label] = publisher_stats[pub].get(label, 0) + 1
    
    for pub in sorted(publisher_stats.keys()):
        print(f"\n{pub}:")
        for label, count in sorted(publisher_stats[pub].items()):
            print(f"  {label}: {count}")

    # Optionally consume input: remove articles that are labeled (not ERROR/SKIPPED) from input JSON
    if consume:
        try:
            with open(json_filepath, "r", encoding="utf-8") as f:
                original = json.load(f)
        except Exception:
            original = {}
        # Build set of labeled URLs including new results
        newly_labeled_urls = set(
            (r.get("article_url") or "")
            for r in results
            if (r.get("label") or "").upper() not in {"ERROR", "SKIPPED"} and (r.get("article_url") or "")
        )
        labeled_urls_all = already_labeled_urls.union(newly_labeled_urls)

        remaining: dict = {}
        if isinstance(original, dict):
            for pub, items in original.items():
                if not isinstance(items, list):
                    continue
                keep = []
                for it in items:
                    url = (it.get("article_url") or "").strip()
                    # Keep items that are not yet labeled OR have no URL
                    if not url or url not in labeled_urls_all:
                        keep.append(it)
                if keep:
                    remaining[pub] = keep
        # Overwrite input file with remaining items
        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(remaining, f, indent=2, ensure_ascii=False)
            print(f"\nInput consumed: wrote remaining unlabeled articles back to {json_filepath}")
        except Exception as e:
            print(f"Warning: Failed to write consumed input to {json_filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify news articles for bias toward/against the PH government.")
    parser.add_argument("--input", dest="json_file", default=DEFAULT_INPUT_JSON, help="Path to input JSON file (or set INPUT_JSON in .env)")
    parser.add_argument("--rules", dest="rules_file", default=DEFAULT_RULES_FILE, help="Path to annotation rules text file (or set RULES_FILE in .env)")
    parser.add_argument("--output", dest="output_file", default=DEFAULT_OUTPUT_JSON, help="Path to write labeled results JSON (or set OUTPUT_JSON in .env)")
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Optional sample size (or set SAMPLE_SIZE in .env). Picks diverse sample across publishers.")
    parser.add_argument("--consume", dest="consume", action="store_true", default=CONSUME_INPUT_DEFAULT, help="If set (or CONSUME_INPUT=true in .env), remove already-labeled articles from the input JSON after run.")
    parser.add_argument("--model", dest="model", default="gpt-5-mini", help="OpenAI model to use")

    args = parser.parse_args()

    # Run a quick environment check for Windows users
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!\n"
              "Set it in Windows PowerShell like this, replacing with your key:\n"
              "  $env:OPENAI_API_KEY=\"sk-...\"\n")
        exit(1)

    # Override classify_article default model if user provided --model
    _orig_classify = classify_article
    def _classify_with_model(article_text: str, rules: str, model: str = args.model) -> Optional[Dict[str, str]]:
        return _orig_classify(article_text, rules, model=model)
    classify_article = _classify_with_model  # type: ignore

    main(args.json_file, args.rules_file, args.output_file, args.sample_size, args.consume)

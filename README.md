# News Annotator

Classify Philippine government-related news articles into one of five bias categories using the OpenAI API and your annotation rules. The script now supports .env-driven defaults, structured outputs, append-with-dedup, diverse sampling, and optional input consumption.

## Setup (Windows PowerShell)

1. (Optional) Create a virtual environment

```powershell
python -m venv .venv; . .\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file (recommended)

Create a file named `.env` in the project folder with at least your API key. You can also set paths and behavior here.

```
OPENAI_API_KEY=sk-your-key

# Optional: default paths and behavior
INPUT_JSON=sampled_articles.json
RULES_FILE=annotation_rules_enhanced.txt
OUTPUT_JSON=labeled_results.json
SAMPLE_SIZE=5           # omit or leave empty to process all
CONSUME_INPUT=true      # remove already-labeled items from input after run
```

The script loads this automatically via `python-dotenv`. If you prefer a temporary key for the current window, you can run:

```powershell
$env:OPENAI_API_KEY="sk-your-key"
```

## Run

With `.env` in place, you can simply run:

```powershell
python classify_articles.py
```

You can also override `.env` defaults with CLI flags:

```powershell
python classify_articles.py ^
  --input sampled_articles.json ^
  --rules annotation_rules_enhanced.txt ^
  --output labeled_results.json ^
  --sample-size 10 ^
  --consume
```

Arguments:

- `--input` (default: from `INPUT_JSON` or `sampled_articles.json`) – Path to input JSON.
- `--rules` (default: from `RULES_FILE` or `annotation_rules.txt`) – Path to the text rules file (you can use `annotation_rules_enhanced.txt`).
- `--output` (default: from `OUTPUT_JSON` or `labeled_results.json`) – Path for labeled results; appends and de-duplicates by `article_url` per publisher.
- `--sample-size` (default: from `SAMPLE_SIZE` or all) – Picks a diverse sample across publishers.
- `--consume` (default: from `CONSUME_INPUT=false`) – If set, removes already-labeled articles from the input JSON after the run.
- `--model` (default: `gpt-4o-mini`) – OpenAI chat model to use.

## Behavior and outputs

- Structured fields per article in results:
  - `label` (one of the 5 categories)
  - `primary_marker` (dominant marker/pattern)
  - `reasoning` (2–3 sentences)
  - `confidence` (High/Medium/Low)
- Appends to existing output and de-duplicates by `article_url` for each publisher (keeps the most recent).
- Skips articles already labeled in prior runs (by `article_url`), avoiding duplicate API calls.
- Optional input “consumption”: with `--consume` or `CONSUME_INPUT=true`, labeled items are removed from the input JSON after the run, leaving only unlabeled items.
- Sampling: when `--sample-size` or `SAMPLE_SIZE` is set, the script selects a diverse subset across publishers (one per publisher first, then round‑robin).
- Very short or empty articles are skipped.
- Basic retries are implemented for transient API errors.

## Input JSON format

The script expects a dictionary keyed by publisher, with each value a list of article objects. Example:

```json
{
  "Some Publisher": [
    {
      "article_url": "https://example.com/a",
      "author_name": "Author",
      "article_headline": "Headline text",
      "date_published": "2025-01-01",
      "article_body": "Full article text...",
      "publisher": "Some Publisher"
    }
  ]
}
```

Results are saved in the same nested-by-publisher shape with additional fields: `label`, `primary_marker`, `reasoning`, and `confidence`.

## Initial Verification

Use verification_sample_articles.json to validate and cross-reference results with manual annotation in order to gauge the accuracy of the model.

A file called verification_articles.json will contain the manually annotated articles, while labeled_verification_results.json will contain the GPT annotated articles to compare.

The categories are: BF, SBF, N, SBA, BA, which are acronyms of the 5 categories for brevity.

## Tips

- To use your enhanced rules by default, set `RULES_FILE=annotation_rules_enhanced.txt` in `.env`.
- If your paths contain spaces, quote them on the CLI, e.g., `--rules "C:\\path with spaces\\annotation_rules_enhanced.txt"`.
- If you prefer not to mutate your input data, leave `CONSUME_INPUT` unset or `false` (default).
- You can always re-run with `--sample-size` for quick spot checks before running the full set.

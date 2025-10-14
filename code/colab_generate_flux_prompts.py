# -*- coding: utf-8 -*-
"""
Colab-ready script: Generates Flux.1-Schnell image prompts ("flux_prompt") and short
explanations ("image_explainer") for each row of an Excel file that follows a fixed schema.

Inputs (required columns):
- "word root"
- "word root meaning"
- "derivative word"
- "derivative explanation by etymonline"
- "OpenAI Explanation"

Outputs:
- Same Excel with two extra columns: "flux_prompt", "image_explainer"
- File name pattern: <input>.xlsx -> <input>_with_flux.xlsx

API:
- Uses Groq's LLM endpoint (chat/completions, JSON mode) for prompt + explainer.
- Read the key from environment variable GROQ_API_KEY.

Author: (c) Pantelis Ladopoulos (Etymologyneering)
License: MIT
"""

# =========================== 0) Minimal deps ================================
# In Colab, you may need (uncomment if necessary):
# !pip install requests pandas openpyxl

import os, re, json, time, requests
import pandas as pd

# In Colab only: use file picker and later auto-download
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# =========================== 1) User I/O ====================================
def pick_excel_filename() -> str:
    """
    If we're in Colab, open a file picker. Otherwise, assume the user placed the
    file next to this script and return the first .xlsx we find (or raise).
    """
    if IN_COLAB:
        print("📁 Upload your .xlsx…")
        uploaded = files.upload()  # opens file picker
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        fname = list(uploaded.keys())[0]
        return fname
    else:
        xlsxs = [f for f in os.listdir(".") if f.lower().endswith(".xlsx")]
        if not xlsxs:
            raise FileNotFoundError("No .xlsx file found in current folder.")
        print(f"Using local file: {xlsxs[0]}")
        return xlsxs[0]

# =========================== 2) Config =====================================
API_KEY   = os.getenv("GROQ_API_KEY", "").strip()    # set via environment
MODEL_ID  = "meta-llama/llama-4-scout-17b-16e-instruct"
API_URL   = "https://api.groq.com/openai/v1/chat/completions"

# Run behavior
FORCE_REDO          = False          # if True, overwrite existing flux_prompt
CHECKPOINT_EVERY    = 5              # save every N rows
SLEEP_BETWEEN_CALLS = 1.2
MAX_RETRIES_429     = 5
BASE_BACKOFF        = 2.0

# Required columns
REQUIRED_COLS = {
    "word root",
    "word root meaning",
    "derivative word",
    "derivative explanation by etymonline",
    "OpenAI Explanation",
}

# System prompts (concise, JSON-only)
SYNTH_SYS = (
    'You are an expert visual concept designer for image generation. '
    'Return strict JSON only: {"prompt":"..."} . '
    'Write ONE concrete scene ≤55 words, 1–2 sentences. '
    'Goal: depict the derivative word’s MODERN MEANING clearly with strong subject, setting, action. '
    'Avoid on-canvas text/logos; keep background simple; end with an art finish + palette.'
)

EXPLAIN_SYS = (
    'You write concise, concrete explanations. '
    'Return strict JSON only: {"how_it_captures_meaning":"...", "stem_visual_mapping":"..."} . '
    'Together ≤70 words. Line 1: how the scene embodies the modern meaning. '
    'Line 2: ONLY IF a PIE/root is provided, name the root and one visual that could symbolize it; else "N/A".'
)

# =========================== 3) Helpers ====================================
def word_cap(s: str, n: int = 55) -> str:
    """Keep at most n words (hard cap for text-to-image prompts)."""
    return " ".join((s or "").split()[:n])

def first_sentence(text: str, max_words: int = 30) -> str:
    """Extract the first sentence as a compact 'modern meaning'."""
    if not isinstance(text, str) or not text.strip():
        return ""
    sent = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return " ".join(sent.split()[:max_words])

def shorten_block(block: str, max_chars: int = 1600) -> str:
    """Trim long context blocks and remove trailing example/conclusion sections."""
    if not isinstance(block, str):
        return ""
    block = re.split(r"\n\s*Example sentences?:", block, flags=re.IGNORECASE)[0]
    block = re.split(r"\n\s*Conclusion:", block, flags=re.IGNORECASE)[0]
    block = block.strip()
    if len(block) > max_chars:
        block = block[:max_chars].rsplit(" ", 1)[0]
    return block

def local_prompt_from_meaning(meaning: str) -> str:
    """Fallback prompt if the API fails—keeps the pipeline moving."""
    meaning_l = (meaning or "").lower()
    if any(k in meaning_l for k in ["humiliation", "hopeless", "degradation", "wretched"]):
        base = "A lone figure on a low step in a dim, empty alley, shoulders slumped."
    elif any(k in meaning_l for k in ["proximity", "near", "adjacent", "nearness"]):
        base = "Two buildings close together, a narrow passage between them, people passing by."
    else:
        base = "A single clear subject doing one action in a minimal background."
    return word_cap(f"{base} Watercolor & ink, muted tones on light background.", 55)

def call_groq_json(system_msg: str, user_msg: str,
                   max_tokens: int = 300, temp: float = 0.2, timeout: int = 45) -> dict:
    """
    Robust JSON-mode call to Groq with simple 429 backoff.
    Raises RequestException on repeated failure.
    """
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Export it in your environment.")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        "response_format": {"type": "json_object"},
        "temperature": temp,
        "max_tokens": max_tokens,
    }

    time.sleep(SLEEP_BETWEEN_CALLS)
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            # Handle rate limit
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else BASE_BACKOFF * (2 ** (attempt - 1))
                time.sleep(min(wait_s, 30))
                if attempt <= MAX_RETRIES_429:
                    payload["messages"].append({"role": "system", "content": "Return strict JSON only. No prose."})
                    continue
                raise RuntimeError(f"429 Too Many Requests after {attempt-1} retries: {r.text[:200]}")
            # Surface bad request early (helps debugging)
            if r.status_code == 400:
                raise RuntimeError(f"400 Bad Request: {r.text[:300]}")
            r.raise_for_status()

            txt = r.json()["choices"][0]["message"]["content"].strip()
            # Some models wrap JSON in ```…```
            if txt.startswith("```"):
                txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(txt)

        except requests.RequestException as e:
            if attempt <= MAX_RETRIES_429:
                time.sleep(BASE_BACKOFF * (2 ** (attempt - 1)))
                continue
            raise e

# =========================== 4) Main ========================================
def main():
    # ---- A) Get file names ----
    fname = pick_excel_filename()
    out_path = fname.replace(".xlsx", "_with_flux.xlsx")

    # ---- B) Load and validate schema ----
    df = pd.read_excel(fname)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure output columns exist
    for col in ("flux_prompt", "image_explainer"):
        if col not in df.columns:
            df[col] = ""

    GENERIC_EXPLAINER = "1) The central action and setting convey the modern meaning.  2) N/A"
    needs_fix = []

    total = len(df)
    processed = 0
    last_saved = 0

    # ---- C) Iterate rows ----
    for i, row in df.iterrows():
        if not FORCE_REDO and str(row.get("flux_prompt", "")).strip():
            processed += 1
            print(f"Processed {processed}/{total} (skip row {i}: already filled)")
            continue

        word   = (row.get("derivative word", "") or "").strip()
        pie    = (row.get("word root", "") or "").strip()
        pie    = pie if pie.startswith("*") else ""  # keep only PIE-looking forms
        oa_exp = (row.get("OpenAI Explanation", "") or "").strip()
        ety    = (row.get("derivative explanation by etymonline", "") or "").strip()

        # Meaning & context source
        meaning = first_sentence(oa_exp) or first_sentence(ety)
        block   = oa_exp if oa_exp else ety

        if not block.strip():
            df.at[i, "flux_prompt"]     = "[SKIPPED: empty OpenAI & etymonline]"
            df.at[i, "image_explainer"] = "[SKIPPED: empty OpenAI & etymonline]"
            processed += 1
            print(f"Processed {processed}/{total} (row {i}: empty sources)")
            if processed - last_saved >= CHECKPOINT_EVERY:
                df.to_excel(out_path, index=False); last_saved = processed
                print(f"Checkpoint saved at {processed} → {out_path}")
            continue

        # ---- C1) Prompt generation (API with graceful fallbacks) ----
        short_block = shorten_block(block, max_chars=1600)
        synth_user = (
            f"Derivative word: {word}\n"
            f"Modern meaning: {meaning}\n"
            f"Source (shortened):\n\"\"\"\n{short_block}\n\"\"\""
        )

        try:
            synth  = call_groq_json(SYNTH_SYS, synth_user, max_tokens=220, temp=0.2)
            prompt = word_cap(synth.get("prompt", ""), 55)
        except Exception:
            # Lightweight fallback using only the meaning
            try:
                synth2 = call_groq_json(
                    SYNTH_SYS,
                    f"Derivative word: {word}\nModern meaning: {meaning}\n",
                    max_tokens=200, temp=0.2
                )
                prompt = word_cap(synth2.get("prompt", ""), 55)
            except Exception:
                prompt = local_prompt_from_meaning(meaning)

        df.at[i, "flux_prompt"] = prompt

        # ---- C2) Explainer generation ----
        try:
            explain_user = (
                f"Derivative word: {word}\n"
                f"Modern meaning: {meaning}\n"
                f"PIE root (if any): {pie}\n"
                f"Text-to-image prompt used:\n\"\"\"\n{prompt}\n\"\"\""
            )
            exp   = call_groq_json(EXPLAIN_SYS, explain_user, max_tokens=180, temp=0.2)
            line1 = (exp.get("how_it_captures_meaning", "") or "").strip()
            line2 = (exp.get("stem_visual_mapping", "") or "").strip()
            if line2.strip().upper() in {"N/", "NA", "N.A", "N-A", ""}:
                line2 = "N/A"
            expl = f"1) {line1}  2) {line2}"
        except Exception:
            expl = GENERIC_EXPLAINER
            needs_fix.append((i, word, meaning, pie, prompt))

        df.at[i, "image_explainer"] = expl

        # ---- C3) Progress + checkpoints ----
        processed += 1
        print(f"Processed {processed}/{total} (row {i})")
        if processed - last_saved >= CHECKPOINT_EVERY:
            df.to_excel(out_path, index=False); last_saved = processed
            print(f"Checkpoint saved at {processed} → {out_path}")

    # ---- D) Post-pass: retry weak explainers once ----
    if needs_fix:
        print(f"\nPost-pass: retrying {len(needs_fix)} generic explainers…")
        for (i, word, meaning, pie, prompt) in needs_fix:
            try:
                time.sleep(max(SLEEP_BETWEEN_CALLS, 2.0))
                explain_user = (
                    f"Derivative word: {word}\n"
                    f"Modern meaning: {meaning}\n"
                    f"PIE root (if any): {pie}\n"
                    "Write two specific, concrete lines. Avoid generic wording like 'the scene illustrates'.\n"
                    f"Text-to-image prompt used:\n\"\"\"\n{prompt}\n\"\"\""
                )
                exp   = call_groq_json(EXPLAIN_SYS, explain_user, max_tokens=180, temp=0.2)
                line1 = (exp.get("how_it_captures_meaning", "") or "").strip()
                line2 = (exp.get("stem_visual_mapping", "") or "").strip()
                if line2.strip().upper() in {"N/", "NA", "N.A", "N-A", ""}:
                    line2 = "N/A"
                new_expl = f"1) {line1}  2) {line2}"
                # Only replace if it isn't the generic sentence
                if "central action and setting" not in new_expl:
                    df.at[i, "image_explainer"] = new_expl
            finally:
                df.to_excel(out_path, index=False)

    # ---- E) Save + download (Colab) ----
    df.to_excel(out_path, index=False)
    print(f"\n✅ Done. Processed {processed}/{total} rows → {out_path}")
    if IN_COLAB:
        try:
            files.download(out_path)  # type: ignore
        except Exception:
            pass

if __name__ == "__main__":
    main()

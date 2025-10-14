# -*- coding: utf-8 -*-
"""
Colab script: generate images with FLUX.1-schnell from prompts in an Excel file.

Expected columns (case/spacing tolerant):
- "flux_prompt"      (text-to-image prompt)
- "derivative word"  (used for filename)

What it does:
1) Lets you upload the Excel from Colab.
2) Reads the two columns, skips empty prompts.
3) Calls Hugging Face Inference API (FLUX.1-schnell) to create JPGs.
4) Saves a CSV log and optionally zips+downloads all images.

Auth:
- Put your token in HF_TOKEN below OR set an env var: os.environ["HF_TOKEN"].

Dependencies (Colab):
!pip -q install huggingface_hub pandas openpyxl tqdm pillow
"""

# ---------- Install (uncomment in Colab if needed) ----------
# !pip -q install huggingface_hub pandas openpyxl tqdm pillow

import os, re, io, time
import pandas as pd
from collections import defaultdict

# Colab helpers (works locally too)
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

from tqdm import tqdm
from PIL import Image
from huggingface_hub import InferenceClient

# ================= User settings =================
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE").strip()  # <— replace or set env var
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Image generation params (tweak for speed/cost)
IMG_SIZE = "1024x1024"   # e.g., "768x768", "512x512"
GUIDANCE_SCALE = 3.5
STEPS = 10               # try 8–12 depending on provider limits

# Limits & output
MAX_IMAGES = None        # e.g., 20 for a quick test
SAVE_TO_DRIVE = False    # if True, save under Google Drive
OUTPUT_DIR = "/content/flux_images"  # will be created if it doesn't exist

# Column aliases (be lenient with headers)
COLUMN_ALIASES = {
    "flux_prompt": ["flux_prompt", "Flux_Prompt", "Prompt", "flux prompt"],
    "derivative_word": ["derivative word", "Derivative word", "Derivative Word",
                        "derivative_word", "word", "lemma"]
}

# ================= Helpers =================
def find_column(df: pd.DataFrame, candidates):
    """Return the first existing column matching any of the candidate names (case/space tolerant)."""
    norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    raise KeyError(f"None of these columns were found: {candidates}")

def sanitize_filename(name: str) -> str:
    """Make a safe file name from a word/lemma."""
    base = (name or "").strip().lower()
    base = re.sub(r"[^\w\s.-]", "", base, flags=re.ASCII)
    base = re.sub(r"\s+", "_", base)
    return base or "untitled"

def parse_size(sz: str):
    """Parse 'WxH' strings into ints."""
    try:
        w, h = sz.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024

WIDTH, HEIGHT = parse_size(IMG_SIZE)

def ensure_output_dir():
    """Create output folder; mount Drive if requested."""
    global OUTPUT_DIR
    if SAVE_TO_DRIVE and IN_COLAB:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=True)
        OUTPUT_DIR = "/content/drive/MyDrive/flux_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HF client =================
def build_client() -> InferenceClient:
    if not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        raise RuntimeError("Set HF_TOKEN (string above) or export env var HF_TOKEN.")
    return InferenceClient(provider="auto", api_key=HF_TOKEN)

def text2image(client: InferenceClient, prompt: str) -> Image.Image:
    """
    Try different argument styles (providers vary).
    Returns a PIL.Image.
    """
    base_kwargs = dict(model=MODEL_ID, guidance_scale=GUIDANCE_SCALE)

    # 1) size + steps (newer API)
    try:
        img = client.text_to_image(prompt, size=IMG_SIZE, steps=STEPS, **base_kwargs)
        return img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
    except TypeError:
        pass
    except Exception as e:
        if "steps must be" not in str(e):
            raise

    # 2) width/height + steps
    try:
        img = client.text_to_image(prompt, width=WIDTH, height=HEIGHT, steps=STEPS, **base_kwargs)
        return img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
    except TypeError:
        pass
    except Exception as e:
        if "steps must be" in str(e):
            # If your provider imposes a different steps range, lower STEPS (8–10 often works).
            raise
        else:
            raise

    # 3) width/height + num_inference_steps (older API)
    img = client.text_to_image(
        prompt, width=WIDTH, height=HEIGHT, num_inference_steps=STEPS, **base_kwargs
    )
    return img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))

# ================= Main =================
def main():
    # Upload the Excel in Colab
    if IN_COLAB:
        print("📤 Upload your .xlsx…")
        uploaded = files.upload()  # type: ignore
        if not uploaded:
            raise SystemExit("No file uploaded.")
        input_xlsx = list(uploaded.keys())[0]
    else:
        # Local fallback: use first .xlsx found
        xlsxs = [f for f in os.listdir(".") if f.lower().endswith(".xlsx")]
        if not xlsxs:
            raise SystemExit("Place an .xlsx next to this script.")
        input_xlsx = xlsxs[0]

    print("✅ Found:", input_xlsx)

    # Read data and pick columns
    df = pd.read_excel(input_xlsx)
    col_prompt = find_column(df, COLUMN_ALIASES["flux_prompt"])
    col_name   = find_column(df, COLUMN_ALIASES["derivative_word"])
    df = df[[col_prompt, col_name]].dropna()
    df = df[df[col_prompt].astype(str).str.strip() != ""]
    print("📄 Rows to process:", len(df))
    print("Prompt column:", col_prompt, "— Name column:", col_name)

    # Prepare output
    ensure_output_dir()
    client = build_client()
    print("✅ HF client ready with model:", MODEL_ID)
    name_counts = defaultdict(int)

    total = len(df) if MAX_IMAGES is None else min(MAX_IMAGES, len(df))
    print(f"🔄 Will generate up to {total} images.")

    log_rows, errors, saved = [], 0, 0

    for idx, row in tqdm(df.iloc[:total].iterrows(), total=total):
        prompt = str(row[col_prompt]).strip()
        word_name = str(row[col_name]).strip()

        if not prompt:
            log_rows.append({"row": int(idx), "name": word_name, "status": "skipped"})
            continue

        base = sanitize_filename(word_name)
        name_counts[base] += 1
        fname = f"{base}_{name_counts[base]:03d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(out_path):
            log_rows.append({"row": int(idx), "name": word_name, "status": "exists", "path": out_path})
            continue

        ok = False
        for attempt in range(6):
            try:
                img = text2image(client, prompt)
                img.save(out_path, "JPEG", quality=95)
                log_rows.append({"row": int(idx), "name": word_name, "status": "saved", "path": out_path})
                saved += 1
                ok = True
                break
            except Exception as e:
                msg = str(e)
                # Backoff for transient/provider errors
                if any(x in msg for x in ["429", "503", "Rate limit", "temporarily unavailable"]):
                    time.sleep(2 ** attempt)
                    continue
                if "steps must be between" in msg:
                    print("⚠️ Provider limit for steps. Try lowering STEPS (e.g., 8–10).")
                errors += 1
                log_rows.append({"row": int(idx), "name": word_name, "status": "error", "error": msg})
                break

        if not ok and (len(log_rows) == 0 or log_rows[-1].get("status") != "error"):
            errors += 1
            log_rows.append({"row": int(idx), "name": word_name, "status": "error", "error": "max retries"})

    print(f"✅ Done. Saved {saved} images, errors {errors}.")

    # Save log CSV
    log_df = pd.DataFrame(log_rows)
    log_csv = os.path.join(OUTPUT_DIR, "generation_log.csv")
    log_df.to_csv(log_csv, index=False)
    print("📄 Log saved at:", log_csv)

    # Zip + download (if not using Drive)
    if IN_COLAB and not SAVE_TO_DRIVE:
        import shutil
        zip_path = "/content/flux_images.zip"
        shutil.make_archive("/content/flux_images", "zip", OUTPUT_DIR)
        print("⬇️ Download the ZIP with images and log:")
        files.download(zip_path)  # type: ignore
    else:
        print("🟢 Files are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

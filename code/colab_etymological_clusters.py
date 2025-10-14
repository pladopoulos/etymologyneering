# -*- coding: utf-8 -*-
"""
Colab script: build clean “Etymological Cluster” diagrams (concentric bubble maps)
that visualize derivative words around their Proto-Indo-European stems.

Input: Excel with columns (case-insensitive):
- "word root"          (stem key used for grouping and the center bubble)
- "word root meaning"  (displayed under the central title)
- "derivative word"    (each becomes a surrounding bubble)

What it does:
1) (Colab) Prompts you to upload the Excel, or uses the first .xlsx in the folder (local).
2) Groups by "word root" and draws one PNG per stem using a clean classic palette.
3) Saves all images under `etymological_clusters/` and creates a ZIP file for download (Colab).

Dependencies (Colab):
!pip -q install pandas matplotlib openpyxl
"""

import os, math, textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import patheffects

# Colab helper (the script also works locally)
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# ----------------- Style / layout -----------------
FIGSIZE = (12, 12)  # inches
DPI     = 300       # print quality

CENTER_R = 0.45
BUBBLE_R = 0.32
R0       = 2.0
DR       = 1.2
SPACING  = 0.25

# Classic palette
COLOR_CENTER = (0.545, 0.0,   0.0  )  # dark red
COLOR_WORD   = (0.96,  0.87,  0.70 )  # wheat
COLOR_RAY    = (0.42,  0.31,  0.19 )  # brown
COLOR_BG     = (1.00,  0.98,  0.94 )  # parchment

CENTER_FS      = 12
WORD_FS_MAX    = 12
WORD_FS_MIN    = 7
WORD_WRAP_BASE = 12

# ----------------- Helpers -----------------
def _wrap(s, w):
    return "\n".join(textwrap.wrap(str(s), width=w)) if s else ""

def ring_capacity(radius, bubble_r=BUBBLE_R, spacing=SPACING):
    """Compute how many bubbles fit around the ring with the given spacing."""
    circumference = 2 * math.pi * radius
    slot = 2 * bubble_r + spacing
    return max(1, int(circumference // slot))

def concentric_layout(words):
    """Arrange words into rings with evenly spaced angles."""
    remaining = list(words)
    rings = []
    i = 0
    while remaining:
        r = R0 + i * DR
        cap = ring_capacity(r)
        chunk = remaining[:cap]
        remaining = remaining[cap:]
        n = len(chunk)
        angles = [2 * math.pi * k / n for k in range(n)] if n else []
        rings.append((r, list(zip(angles, chunk))))
        i += 1
    return rings

def text_with_stroke(ax, x, y, s, fs, color='black', weight='bold'):
    """Render text with a white outline for readability."""
    t = ax.text(
        x, y, s, fontsize=fs, color=color, ha='center', va='center',
        fontweight=weight, zorder=4, linespacing=1.0
    )
    t.set_path_effects([patheffects.withStroke(linewidth=3.0, foreground='white')])
    return t

def fit_inside_circle(ax, x, y, text, radius,
                      fs_max=WORD_FS_MAX, fs_min=WORD_FS_MIN, wrap_base=WORD_WRAP_BASE):
    """Decrease font size and wrapping width until text fits inside a circle."""
    wrapped, fs = str(text), fs_max
    for extra in range(0, 10):
        wrap = max(6, wrap_base - extra)
        fs = fs_max
        for _ in range(20):
            wrapped = _wrap(text, wrap)
            t = text_with_stroke(ax, x, y, wrapped, fs)
            plt.draw()
            bbox = t.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            inv = ax.transData.inverted()
            (x0, y0) = inv.transform((bbox.x0, bbox.y0))
            (x1, y1) = inv.transform((bbox.x1, bbox.y1))
            if max(abs(x1 - x0), abs(y1 - y0)) <= 2 * radius * 0.95:
                return t
            t.remove()
            if fs > fs_min:
                fs -= 0.5
            else:
                break
    return text_with_stroke(ax, x, y, wrapped, fs_min)

def draw_cluster(center_text, words, title=None, outfile=None):
    """Draw one etymological cluster diagram."""
    rings = concentric_layout(words)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, pad=16, color=COLOR_RAY)

    # Center circle
    ax.add_patch(Circle((0, 0), CENTER_R, facecolor=COLOR_CENTER,
                        edgecolor='black', linewidth=1.2, zorder=2))
    text_with_stroke(ax, 0, 0, _wrap(center_text, 28), fs=CENTER_FS)

    # Rays + word bubbles
    for r, items in rings:
        for ang, word in items:
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            ax.plot([0, x], [0, y], color=COLOR_RAY, linewidth=1.0, zorder=1)
            ax.add_patch(Circle((x, y), BUBBLE_R, facecolor=COLOR_WORD,
                                edgecolor='black', linewidth=1.2, zorder=3))
            fit_inside_circle(ax, x, y, word, radius=BUBBLE_R)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_from_excel(excel_path, sheet_name=0,
                    group_by="word root",
                    col_meaning="word root meaning",
                    col_word="derivative word",
                    out_dir="etymological_clusters"):
    """Read Excel and generate one PNG per stem (Etymological Cluster)."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [c.strip().lower() for c in df.columns]

    for need in [group_by, col_meaning, col_word]:
        if need not in df.columns:
            raise ValueError(f"Missing column '{need}'. Found: {df.columns.tolist()}")

    for stem, g in df.groupby(group_by):
        center = f"‘{stem}’ ({g[col_meaning].iloc[0]})"
        words  = [str(w).strip() for w in g[col_word].dropna().tolist() if str(w).strip()]
        if not words:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in str(stem))[:90]
        outfile = Path(out_dir) / f"{safe}.png"
        draw_cluster(center, words, outfile=str(outfile))
        print("Saved:", outfile)

# ----------------- Entry point -----------------
def main():
    # 1) Upload or detect Excel file
    if IN_COLAB:
        print("📤 Upload your .xlsx file…")
        uploaded = files.upload()  # type: ignore
        if not uploaded:
            raise SystemExit("No file uploaded.")
        excel_path = list(uploaded.keys())[0]
    else:
        xlsxs = [f for f in os.listdir(".") if f.lower().endswith(".xlsx")]
        if not xlsxs:
            raise SystemExit("Place an .xlsx file next to this script.")
        excel_path = xlsxs[0]
    print("✅ Found:", excel_path)

    # 2) Generate Etymological Clusters
    plot_from_excel(
        excel_path,
        sheet_name=0,
        group_by="word root",
        col_meaning="word root meaning",
        col_word="derivative word",
        out_dir="etymological_clusters"
    )
    print("✅ Done. Images saved in folder: etymological_clusters")

    # 3) Zip + download (Colab only)
    if IN_COLAB:
        from zipfile import ZipFile
        import pathlib
        zip_path = "etymological_clusters.zip"
        with ZipFile(zip_path, "w") as zf:
            for p in pathlib.Path("etymological_clusters").glob("*.png"):
                zf.write(p, arcname=p.name)
        print("⬇️ Downloading ZIP file…")
        files.download(zip_path)  # type: ignore

if __name__ == "__main__":
    main()


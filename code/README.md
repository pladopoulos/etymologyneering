# 💻 Code Directory

This folder contains the Python scripts used to generate, analyze, and visualize the **Etymologyneering** materials.

### Scripts

- **`scrape_etymonline.py`** – Extracts etymological data from online sources (Etymonline, Wiktionary) for Proto-Indo-European stems and derivative words.  
- **`flux_image_prompts.py`** – Generates text-to-image prompts and captions for the Flux.1-Schnell model via Hugging Face API.  
- **`visualize_clusters.py`** – Uses Matplotlib to build semantic clusters of derivative words around their PIE stems.

### Requirements
These scripts rely on:
```bash
pip install requests beautifulsoup4 matplotlib openai

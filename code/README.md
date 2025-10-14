# 💻 Code Directory

This folder contains the Python scripts used to generate, analyze, and visualize the **Etymologyneering** materials.

### Scripts

- **`colab_generate_flux_prompts.py`** – Extracts etymological data from online sources (Etymonline, Wiktionary) for Proto-Indo-European stems and derivative words.  
- **`colab_flux_HF_image_generator.py`** – Generates text-to-image prompts and captions for the Flux.1-Schnell model via Hugging Face API.  
- **`colab_etymological_clusters.py`** – Uses Matplotlib to build semantic clusters of derivative words around their PIE stems.

### Requirements
These scripts rely on:
```bash
pip install requests beautifulsoup4 matplotlib openai

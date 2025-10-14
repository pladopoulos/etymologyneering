# 💻 Code Directory

This folder contains the Python scripts used to generate, analyze, and visualize the **Etymologyneering** materials.

### Scripts

- **`colab_generate_flux_prompts.py`** – Generates **text-to-image prompts** and concise **visual explainers** using Groq’s LLaMA-4 model.  
Output columns: `flux_prompt`, `image_explainer`.
- **`colab_flux_HF_image_generator.py`** – Generates actual **images** from the `flux_prompt` column in your Excel file,  
using the **black-forest-labs/FLUX.1-Schnell** model hosted on **Hugging Face**.  
Each derivative word is converted into a rendered scene saved as `.jpg` files.  
- **`colab_etymological_clusters.py`** – Builds **Etymological Cluster Diagrams** — clean concentric “bubble maps”  
that visualize **derivative words** orbiting their **Proto-Indo-European stems**.  
Uses **Matplotlib** to generate one `.png` per root (cluster).  

### Requirements
These scripts rely on:
```bash
pip install requests beautifulsoup4 matplotlib openai

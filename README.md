# ml-narrative-pace-style-analysis

Scripts and code for machine learning-based analysis of **narrative pace** and **stylistic features** in *Virginia Woolf’s To the Lighthouse*, enabling reproducible computational literary research.

---

## Overview
This repository contains the **analysis scripts** for a machine learning–enhanced computational study of narrative pace and stylistic features in *Virginia Woolf’s To the Lighthouse*.  
The framework integrates NLP tools such as **spaCy**, **Sentence-BERT**, and **WordNet** to quantify intra-textual dynamics and reveal patterns in **psychological temporality** and **modernist style**.  
It includes scripts for text segmentation, feature extraction, statistical analysis, and visualization, ensuring full methodological transparency and reproducibility.

This repository accompanies the paper:  
**“Machine learning-enhanced end-to-end computational methods for the analysis of narrative pace and stylistic features in Virginia Woolf’s *To the Lighthouse*.”**

**Note:**  
To maintain the integrity of the peer review process, all raw and processed data are withheld during review.  
Upon acceptance, both datasets will be released through **Dryad** and linked here.  
All analysis scripts, environment settings, and processing logic are fully available in this repository.

---

## Repository Structure
```
ml-narrative-pace-style-analysis/
├── text_preprocessing.py             # Text segmentation and preprocessing
├── narrative_pace_analysis.py        # Narrative pace feature extraction and analysis
├── stylistic_features_analysis.py    # Stylistic feature extraction and analysis
├── data/                             # (empty) Placeholder for source text
│ └── .gitkeep
├── Narrative_Analysis_output/        # (empty) Placeholder for generated results
│ └── .gitkeep
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Python-specific ignores
├── LICENSE                           # MIT License
└── README.md                         # This file
```
---

## Installation

### Prerequisites
- Python 3.10 or above  
- Git (for cloning the repository)

### Setup
git clone https://github.com/BiLiqi777/ml-narrative-pace-style-analysis.git
cd ml-narrative-pace-style-analysis
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

---

## Usage

### 1. Text Preprocessing
Segments the novel into parts, chapters, and paragraphs, and prepares data for analysis.
```
python text_preprocessing.py
```
### 2. Narrative Pace Analysis
Extracts narrative pace features (e.g., event density, perspective shift, temporal density), performs statistical tests, and generates visualizations.
```
python narrative_pace_analysis.py
```
### 3. Stylistic Feature Analysis
Computes stylistic features (e.g., syntactic complexity, repetition, sentiment), conducts clustering, and outputs visualizations.
```
python stylistic_features_analysis.py
```
---

## Data Availability
The novel text and all processed data are currently withheld during peer review to ensure double-blind integrity.
Upon acceptance, both the raw and processed datasets will be made publicly available via Dryad, and this section will be updated with the DOI.

---

## Citation
If you use or adapt the analysis scripts, please cite:
```
@article{pan_bi_2025,
  title = {Machine learning-enhanced end-to-end computational methods for the analysis of narrative pace and stylistic features in Virginia Woolf’s To the Lighthouse},
  author = {Pan, Yan and Bi, Liqi},
  year = {2025},
  note = {Code available at: https://github.com/BiLiqi777/ml-narrative-pace-style-analysis}
}
```
---

## License
This project is licensed under the MIT License — see the LICENSE file for details.

---

## Acknowledgments
Tools: spaCy, Sentence-Transformers, UMAP, HDBSCAN, NLTK (VADER), scikit-learn, etc.
Data: Virginia Woolf’s To the Lighthouse (public domain via Project Gutenberg, to be released post-review).

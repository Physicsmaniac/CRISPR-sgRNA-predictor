
# CRISPR sgRNA Predictor

A deep learning-based tool designed to predict and rank single-guide RNA (sgRNA) candidates for CRISPR-Cas9 gene editing applications.

---

## 🚀 Overview

This project leverages a trained neural network model (`crispr_model.h5`) to evaluate and score potential sgRNA sequences based on their predicted efficiency and specificity. The tool aids researchers in selecting optimal sgRNA candidates for targeted gene editing.

---

## 🧬 Features

* Predicts sgRNA efficiency and specificity scores.
* Ranks sgRNA candidates based on model predictions.
* Supports input sequences for genes like *lacZ* and *rpoB*.
* Provides insights into GC-content and error correlations.

---

## 🛠️ Installation

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/Physicsmaniac/CRISPR-sgRNA-predictor.git](https://github.com/Physicsmaniac/CRISPR-sgRNA-predictor](https://github.com/Physicsmaniac/CRISPR-sgRNA-predictor)
   cd CRISPR-sgRNA-predictor
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

LOGINTEL: AI-ENABLED THREAT LOG ANALYZER
README.txt

--------------------------------------------------------------------------------
1. Project Overview
--------------------------------------------------------------------------------
This repository implements a semantics-aware, adversarially robust log anomaly detection pipeline,
with a suite of standalone experiments for evaluating different modeling approaches.
Core pipeline steps:
  • Feature engineering
  • Adversarial training
  • GRU-based sequence classification

--------------------------------------------------------------------------------
2. Directory Structure
--------------------------------------------------------------------------------
├── main/  
│   ├── Adversarial Training.py  
│   │     ▸ Synthesizes adversarial log samples and trains defense ensemble  
│   ├── Feature Eng.py  
│   │     ▸ Extracts statistical, semantic, and topic‐modeling features from embedded logs  
│   └── GRU classification.py  
│         ▸ Defines & trains the GRU sequence detector; outputs performance metrics  
│  
├── Experiments/  
│   ├── distilroberta.py  
│   │     ▸ Baseline experiments with DistilRoBERTa embeddings  
│   ├── electra_2_0.py  
│   │     ▸ Baseline experiments using ELECTRA embeddings  
│   ├── knowledge_graph_embedding.py  
│   │     ▸ Tests log representation via knowledge-graph embeddings  
│   ├── Topic_modelling_lda features.py  
│   │     ▸ Feature extraction via LDA topic modeling  
│   ├── week_3.py  
│   │     ▸ Initial weekly experiments on new different feature aspects of system logs.
│   ├── week3_1.py  
│   │     ▸ Variant of week 3 experiments (parameter sweep)  
│   ├── week4.py  
│   │     ▸ Week 4 experiments: extended feature sets & model comparisons  
│   ├── week5(shap).py  
│   │     ▸ Week 5 explainability: SHAP-based attribution analyses  
│   ├── week5_(semantic_features_loganamoly).py  
│   │     ▸ Week 5 semantic-feature extraction & anomaly tests  
│   └── week5_1.py  
│         ▸ Additional week 5 experiments
│  
├── requirements.txt  
│     ▸ List of Python dependencies  
│  
└── README.txt  
      ▸ This file  

--------------------------------------------------------------------------------
3. Prerequisites
--------------------------------------------------------------------------------
• Python 3.8 or higher  
• (Optional) CUDA-enabled GPU for faster training  
• Install dependencies:
    pip install -r requirements.txt

Key libraries include:
  • torch, torchvision  
  • sentence-transformers  
  • scikit-learn, numpy, pandas  
  • nltk, umap-learn, shap  
  • matplotlib

--------------------------------------------------------------------------------
4. Usage
--------------------------------------------------------------------------------
4.1 Run the full pipeline (main folder)  
    1. Feature extraction:
       python "main/Feature Eng.py" --input <raw_log_dir> --output results/features.pkl  
    2. Adversarial training:
       python "main/Adversarial Training.py" --features results/features.pkl --output results/adv_models/  
    3. GRU classification:
       python "main/GRU classification.py" --features results/features.pkl --models results/adv_models/  

4.2 Run an individual experiment  
    python "Experiments/<script_name>.py" [--config configs.yaml]  
  e.g.:
    python Experiments/distilroberta.py --data results/features.pkl

--------------------------------------------------------------------------------
5. File Descriptions
--------------------------------------------------------------------------------
• **main/Feature Eng.py**  
    - Loads raw logs, embeds with SBERT, extracts time-series & topic features.  

• **main/Adversarial Training.py**  
    - Generates adversarial log variants, trains a robust ensemble.  

• **main/GRU classification.py**  
    - Defines & trains a GRU network on sequential features; evaluates AUC, F1, etc.  

• **Experiments/\*.py**  
    - Self-contained scripts for evaluating specific models or feature sets.  
    - Each begins with a header explaining its purpose, inputs, and outputs.

--------------------------------------------------------------------------------
6. Results & Outputs
--------------------------------------------------------------------------------
After running the pipeline, you will find:  
  • `results/features.pkl` — serialized feature matrix  
  • `results/adv_models/` — adversarial ensemble checkpoints  
  • `results/metrics.csv` — classification metrics (accuracy, F1, ROC-AUC)  
  • `results/plots/` — confusion matrices, ROC curves, UMAP/t-SNE visualizations  

--------------------------------------------------------------------------------
7. Extending the Project
--------------------------------------------------------------------------------
• To add a new feature module or model, drop your `.py` into `main/` and update invocation order.  
• For new experiments, create a script under `Experiments/` following the existing naming & docstring conventions.  
• Adjust hyperparameters or file paths via command-line args or by creating a simple `configs.yaml`.


Questions or contributions?  
— email: gourabmahato09@gmail.com  

# GNN Airfoil Pressure Distribution Prediction

graph neural network surrogate model for predicting airfoil pressure coefficient (Cp) using the airfrans cfd dataset.

## overview

predicts pressure coefficient distributions around 2d airfoils using graph neural networks (gnns) as a fast surrogate for cfd simulations.

**expected outcomes:**
- cp mse < 1e-3 (normalized)
- lift error < 10%
- r² (cp) > 0.9

## quick start

```bash
# 1. clone repository
git clone https://github.com/anaya-rs/GNN-Airfoil-Project.git
cd GNN-Airfoil-Project

# 2. setup environment
python -m venv venv
venv\Scripts\activate  # windows
# source venv/bin/activate  # mac/linux

# 3. install dependencies
pip install -r requirements.txt

# 4. preprocess data (requires full 9.3gb download)
python quick_preprocess.py  # quick: 100/50/50 samples
# python data_preprocess.py --quick  # smaller: 50/20/20 samples
# python data_preprocess.py  # full: 500/100/100 samples

# 5. train models
python train_gcn.py
python improved_gnn.py

# 6. evaluate
python evaluate.py
```

## dataset

**⚠️ important:** the airfrans dataset (~9.3gb) must be downloaded in full. there is no way to download only a subset. even "quick" options download the entire dataset first, then process fewer samples.

**to avoid downloading:**
- place `Dataset.zip` in `./airfrans_data/` folder (script will detect it)

**download options:**
- automatic download on first run
- manual download: place `Dataset.zip` in `./airfrans_data/`

## usage

### preprocess data

```bash
python quick_preprocess.py        # 100/50/50 samples (recommended)
python data_preprocess.py --quick # 50/20/20 samples
python data_preprocess.py         # 500/100/100 samples (full)
python check_download_status.py   # check download status
```

### train models

```bash
python train_gcn.py      # baseline gcn model
python improved_gnn.py   # gat and graphsage models
```

### evaluate

```bash
python evaluate.py
```

## project structure

```
.
├── data_preprocess.py          # data preprocessing
├── quick_preprocess.py         # quick preprocessing (100/50/50)
├── check_download_status.py    # check dataset status
├── train_gcn.py                # baseline gcn training
├── improved_gnn.py             # improved models (gat/graphsage)
├── evaluate.py                 # model evaluation
├── requirements.txt            # dependencies
└── README.md
```

## results

results are saved in:
- `./results/` - baseline gcn
- `./results_improved/` - gat model
- `./results_sage/` - graphsage model
- `./evaluation_results_*/` - evaluation results

## requirements

- python 3.8+
- pytorch
- pytorch geometric
- matplotlib, numpy, scikit-learn
- airfrans

## troubleshooting

**out of memory:** use `quick_preprocess.py` or reduce batch size in training scripts

**download issues:** manually download `Dataset.zip` and place in `./airfrans_data/`

**module errors:** ensure virtual environment is activated and dependencies are installed

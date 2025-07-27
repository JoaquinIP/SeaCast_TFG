# Seacast

## 🚀 How to Use
### 1. Clone repository:

```bash
git clone https://github.com/aiis-group/Seacast.git
```
### 2. Create environment:

```bash
conda create -n Seacast_env python=3.10.16
conda activate Seacast_env
```
### 3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🚀 SeaCast Deployment

### 1. Data Download

Before downloading data, you need to set up your CDS API credentials by creating a configuration file at $HOME/.cdsapirc. Instructions for creating this file can be found at:
https://cds.climate.copernicus.eu/how-to-api

Download the required data for the period 2018-01-01 to 2023-12-31:

```bash
user=[CMEMS-username]
password=[CMEMS-password]

python download_data.py --static -b data/atlantic/ -u $user -psw $password &&
python download_data.py -d reanalysis -s 2018-01-01 -e 2023-12-31 -u $user -psw $password &&
python download_data.py -d era5 -s 2018-01-01 -e 2023-12-31
```

### 2. Data Preparation

#### Dataset Splits
- **Training set**: 2018-01-01 to 2021-12-31
- **Validation set**: 2022-01-01 to 2022-12-31
- **Test set**: 2023-01-01 to 2023-12-31

#### 3. Prepare States - Reanalysis Data

```bash
# Training set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/train -n 6 -p rea_data -s 2018-01-01 -e 2021-12-31

# Validation set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/val -n 6 -p rea_data -s 2022-01-01 -e 2022-12-31

# Test set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/test -n 17 -p rea_data -s 2023-01-01 -e 2023-12-31
```

#### 4. Prepare States - ERA5 Data

```bash
# Training set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/train -n 6 -p forcing -s 2018-01-01 -e 2021-12-31

# Validation set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/val -n 6 -p forcing -s 2022-01-01 -e 2022-12-31

# Test set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/test -n 17 -p forcing -s 2023-01-01 -e 2023-12-31
```

### 5. Feature and Model Preparation

```bash
# Create grid features
python create_grid_features.py --dataset atlantic

# Create parameter weights
python create_parameter_weights.py --dataset atlantic --batch_size 4 --n_workers 4

# Create mesh
python create_mesh.py --dataset atlantic --graph hierarchical --levels 3 --hierarchical 1
```

### 6. Model Training

Train the Hi-LAM model with the following configuration:

```bash
python train_model.py --dataset atlantic \
                     --n_nodes 1 \
                     --n_workers 4 \
                     --epochs 100 \
                     --lr 0.001 \
                     --batch_size 1 \
                     --step_length 1 \
                     --ar_steps 1 \
                     --optimizer adamw \
                     --scheduler cosine \
                     --processor_layers 4 \
                     --hidden_dim 128 \
                     --model hi_lam \
                     --graph hierarchical \
                     --finetune_start 1 \
```

### Notes
- Ensure all dependencies are installed before running the commands
- Adjust the number of workers (`--n_workers`) based on your system capabilities
- The batch size (`--batch_size`) can be modified according to available GPU memory
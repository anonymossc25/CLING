---

# CLING
This project is a modified fork of the official [DGL (Deep Graph Library)](https://github.com/dmlc/dgl). We have created a new branch in our fork and integrated our custom code for experiments and performance analysis, particularly targeted at SC 2025 paper artifact regeneration.
Please follow the DGL installation process if you get any Error.

---
## 🖥️ System Configuration

We conducted our experiments on a machine with the following specifications:

- **GPU**: NVIDIA RTX A6000 with 48 GB memory  
- **CPU**: Intel(R) Xeon(R) Gold 5218, 16 cores @ 2.30 GHz  
- **RAM**: 512 GB  
- **Software Stack**:
  - DGL v2.0.0  
  - PyTorch v1.13.0  
  - CUDA 11.7  
  - GCC 9.5.0  

---

## 📦 Installation Process

### Option 1: Install via Conda Environment File

To quickly set up the exact same environment used in our experiments:

```bash
conda env create -f cling.yml
conda activate dgl-gpu
```

Then proceed with the build and install:

```bash
bash script/build_dgl.sh -g
cd python
python setup.py install
python setup.py build_ext --inplace
```

> Make sure you have CUDA 11.7 installed and accessible.

---

### Option 2: Install from Source Manually

## Install from Source

Download the source files from GitHub:

```bash
git clone --recurse-submodules https://github.com/ssecsurendra/CLING.git
cd dgl_cluster
git submodule update --init --recursive
```

### Linux Dependencies

#### For Debian/Ubuntu users:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev make cmake
```

#### For Fedora/RHEL/CentOS users:

```bash
sudo yum install -y gcc-c++ python3-devel make cmake
```

### Conda Environment Setup for GPU Development

```bash
bash script/create_dev_conda_env.sh -g 11.7
```

To check additional configuration options:

```bash
bash script/create_dev_conda_env.sh -h
```

### Build the Shared Library for GPU

```bash
bash script/build_dgl.sh -g
```

For more build options:

```bash
bash script/build_dgl.sh -h
```

### Install Python Bindings

```bash
cd python
python setup.py install
# Build Cython extensions
python setup.py build_ext --inplace
```

---

## 🧪 Regenerate SC 2025 Artifact
### Clean Previously Generated Files

```bash
bash clean.sh
```

---
### Run GraphSAGE and Cling Scripts

```bash
cd $DGL_HOME
bash generate_logs.sh
bash generate_logs_gcn.sh
bash generate_logs_layer.sh
```

This will run both GraphSAGE and Cling across various configurations (datasets, batch sizes, fanouts) and store logs in:

```
logs -> graphsage or cling -> <dataset_name.txt>
```

> ⏳ Note: This process may take 10–12 hours.

## 📊 Reproducing Table and Figures from the Paper

### Generate Table 3
> **Reduction of number of neighbors in each layers on different dataset (Batch size 1024, Fanout 20)**
```bash
bash table_3.sh
```

### Generate Table 4

> **Performance breakdown with fanout 20 and batch size 1024 (in seconds)**

```bash
bash table_4.sh
```

### Generate Figure 7 – Effect of Varying Fanout

```bash
bash gen_Fig_7.sh <dataset_name>
```

Example:

```bash
bash gen_Fig_7.sh reddit
```

### Generate Figure 8 – Effect of Varying Batch Size

```bash
bash gen_Fig_8.sh <dataset_name> <fanout_value>
```

Example:

```bash
bash gen_Fig_8.sh ogbn-products 15
```

### Generate Figure 9 – Effect of Varying Number of Layers

```bash
bash gen_Fig_9.sh <dataset_name>
```

Example:

```bash
bash gen_Fig_9.sh ogbn-products
```

### Generate Figure 10 – Comparison with GCN Aggregation

```bash
bash gen_Fig_10.sh <fanout_value>
```

Example:

```bash
bash gen_Fig_10.sh 15
```

---

## 📌 Notes

- Figures 4, 5, and 6 are derived from data extracted using the **NVIDIA profiler**.
- For accurate reproduction, ensure all logs are generated correctly before running figure or table generation scripts.

---

## 🧬 About This Fork

This project is based on a **fork of the official DGL** repository. We have:

- Created a new working branch.
- Integrated our custom components for large-scale distributed GNN training and performance benchmarking.
- Made significant enhancements tailored to SC 2025 research experiments.

---

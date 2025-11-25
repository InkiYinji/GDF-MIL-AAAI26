# Rethinking Multi-Instance Learning through Graph-Driven Fusion: A Dual-Path Approach to Adaptive Representation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Summary

GDF-MIL (Graph-Driven Fusion Multiple Instance Learning) is a novel graph-driven multi-instance learning framework that adaptively balances topology modeling and semantic feature preservation through a dual-path feature fusion mechanism. The framework significantly outperforms 18 state-of-the-art methods on 24 datasets.

## ✨ Main Features

- **Adaptive bag mapping module (ABMM)**: Maps variable-length packets to compact representations using Gumbel-Softmax soft clustering, significantly reducing computational costs.
- **Dynamic graph structure learning (DGSL)**: Constructs efficient sparse graphs based on weighted connectivity and SAGEConv, fully exploring topological structures.
- **Dual-path feature fusion (DPFF)**: Adaptively fuses graph-level and packet-level features, balancing graph construction efficiency with semantic integrity.

## 🏗️ Project Structure

```
GDF-MIL/
├── configs/              # Configuration file directory
│   ├── GDAMIL.yaml       # GDF-MIL main configuration file
├── datasets/             # Dataset processing module
│   ├── dataloader.py
│   └── MIL.py
├── models/
│   ├── GDAMIL/          # Core
│   │   ├── GDAMIL.py    # Main model
│   │   ├── Attention.py
│   │   ├── gnn.py
│   │   └── ...
│   ├── GAMIL.py 
│   └── process.py
├── utils/
│   ├── basic.py
│   ├── reader.py
│   └── writer.py
├── main.py
└── requirements.txt
```

## 🚀 Quick Start

### Environmental Requirements

- Python >= 3.7
- PyTorch >= 1.8.0

### Installation steps

1. **Cloning repository**
```bash
git clone https://github.com/InkiYinji/GDF-MIL-AAAI26.git
cd GDF-MIL-AAAI26/Code
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Data Preparation

The project supports multiple dataset formats:

1. **MAT Format Dataset** (for benchmark)
   - The dataset should be stored in the specified path as a `.mat` file.
   - - Data structure: `data[i][0]` is the instance feature matrix, `data[i][1]` is the bag label.

2. **H5 Format Dataset** (for WSI)
   - See https://github.com/lingxitong/MIL_BASELINE

### Training

```bash
python main.py
```

### Parameter Description

#### Key Hyperparameters

- **k_components**: Controlling the number of soft cluster centers directly affects the degree of packet compression.
  - Recommended value: {10, 20, 50, 100, 200}
  - Smaller value: Faster computation but may lose information
  - Larger value: Retains more information but increases computational cost

- **k_neighbors**: Number of neighbors during control graph construction
  - Recommended value: ≤ KC，如 {10, 20, 50, 100, 200}
  - The sparsity of graph structure and the range of information aggregation

- **Learning rate**: Default 0.0002

#### Model Architecture Parameters

- **hid_dim**: 256
- **out_dim**: 128
- **dropout**: 0.1

## 📚 Citation

If GDF-MIL is helpful to your research, please cite it.

```bibtex
@inproceedings{Zhang2024gdfmil,
  title={Rethinking multi-instance learning through graph-driven fusion: {A} dual-path approach to adaptive representation},
  author={Zhang, Yu-Xuan and Zhou, Zhengchun and Liu, Weisha and Zhang, Mingxing},
  booktitle={{AAAI}},
  year={2026}
}
```

## 🙏 Acknowledgments

This project is developed based on the [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE) framework. Thanks to the original author for their excellent work.

Special thanks to the following projects:

- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE) - A unified framework for MIL methods
- [CLAM](https://github.com/mahmoodlab/CLAM) - Computational pathology MIL methods


## 📄 License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

## 📧 Contact Information

- **Author**: Yu-Xuan Zhang
- **Email**: inki.yinji@gmail.com

## 🔗 Related Links

- [Data](https://palm.seu.edu.cn/zhangml/)
- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE)

---

**Last Updated: November 25, 2025**

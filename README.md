# Achieving Carbon-Efficient AI

## Comparative Analysis of Model Compression Techniques for Achieving Carbon-Efficient AI

### Authors: Eileen Paula, Jayesh Soni, Himanshu Upadhyay, Leonel Lagos

## Overview
This repository contains the implementation and analysis for our study on reducing energy consumption and carbon emissions in transformer-based models through model compression techniques. The study evaluates the impact of pruning, knowledge distillation, and quantization on the energy efficiency of BERT, DistilBERT, ALBERT, and ELECTRA. The effectiveness of these compression techniques is compared against inherently carbon-efficient models, TinyBERT and MobileBERT.

## Paper
The full research paper can be found [here](https://github.com/eileenpaula/Achieving-Carbon-Efficient-AI).

## Directory Structure
```
├── baseline_models/               # Original transformer models
│   ├── albert.py
│   ├── bert.py
│   ├── distilbert.py
│   ├── electra.py
├── compressed_models/             # Compressed models using pruning, distillation, and quantization
│   ├── albertcomp.py
│   ├── bertcomp.py
│   ├── distilbertcomp.py
│   ├── electracomp.py
├── efficient_models/              # Pre-trained carbon-efficient models
│   ├── mobilebert.py
│   ├── tinybert.py
├── results/                       # Analysis and visualization
│   ├── data_figure.ipynb          # Charts based on dataset
│   ├── results_analysis.ipynb     # Performance and energy consumption analysis using figures
│   ├── statistical_analysis.xlsx  # Results and statistical analysis
├── requirements.txt               # Dependencies required for the project
```

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/eileenpaula/Achieving-Carbon-Efficient-AI.git
   cd Achieving-Carbon-Efficient-AI
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Models
You can run the baseline, compressed, and efficient models using the provided scripts:

### Baseline Models
```sh
python baseline_models/bert.py
python baseline_models/distilbert.py
python baseline_models/albert.py
python baseline_models/electra.py
```

### Compressed Models
```sh
python compressed_models/bertcomp.py
python compressed_models/distilbertcomp.py
python compressed_models/albertcomp.py
python compressed_models/electracomp.py
```

### Efficient Models
```sh
python efficient_models/tinybert.py
python efficient_models/mobilebert.py
```

## Results and Analysis
All results, including energy consumption and performance metrics, are documented in the `results/` directory.

- `data_figure.ipynb`: Visual representation of dataset distribution.
- `results_analysis.ipynb`: Performance metrics and energy consumption analysis.
- `statistical_analysis.xlsx`: Statistical analysis and comparison.

### Key Results
The following table summarizes key performance and energy efficiency results from our experiments:

| Model                          | Accuracy  | Precision | Recall   | F1-score | Energy Consumption (kWh) | CO2 Emissions (kg) |
|--------------------------------|----------|----------|----------|----------|-------------------------|--------------------|
| BERT Baseline                 | 0.96154  | 0.96155  | 0.96154  | 0.96154  | 7.197                   | 3.366              |
| BERT w/ Pruning & Distillation| 0.95900  | 0.95903  | 0.95900  | 0.95900  | 4.887                   | 2.270              |
| DistilBERT Baseline           | 0.95882  | 0.95882  | 0.95882  | 0.95881  | 3.364                   | 1.563              |
| DistilBERT w/ Pruning         | 0.95872  | 0.95872  | 0.95872  | 0.95871  | 3.589                   | 1.667              |
| ALBERT Baseline               | 0.86356  | 0.89430  | 0.86356  | 0.83751  | 7.593                   | 3.528              |
| ALBERT w/ Quantization        | 0.65445  | 0.67820  | 0.65445  | 0.63458  | 7.053                   | 3.276              |
| ELECTRA Baseline              | 0.96699  | 0.96699  | 0.96699  | 0.96699  | 6.607                   | 3.070              |
| ELECTRA w/ Pruning & Distillation | 0.95917  | 0.95918  | 0.95917  | 0.95917  | 5.026                   | 2.335              |
| TinyBERT                      | 0.95154  | 0.95154  | 0.95154  | 0.95154  | 0.629                   | 0.292              |
| MobileBERT                    | 0.92461  | 0.92463  | 0.92461  | 0.92461  | 3.663                   | 1.701              |

## Citation
If you use this repository, please cite our work as follows:
```
@article{paula2025carbon,
  title={Comparative Analysis of Model Compression Techniques for Achieving Carbon-Efficient AI},
  author={Paula, Eileen and Soni, Jayesh and Upadhyay, Himanshu and Lagos, Leonel},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License
This project is licensed under the MIT License.

## Acknowledgment
This research is supported by the U.S. Department of Energy – Environmental Management (DOE-EM) (DE-EM0005213).

## Contact
For any questions or inquiries, please reach out to:
- **Eileen Paula**: eileenkpaula@gmail.com
- **GitHub**: [github.com/eileenpaula](https://github.com/eileenpaula)



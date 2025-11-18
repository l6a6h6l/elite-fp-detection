# ELITE v6.0 - False Positive Detection Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-JCP%202025-green.svg)](https://www.mdpi.com/journal/jcp)

Machine learning framework for detecting false positives in Application Performance Monitoring (APM) systems, with specialized features for Latin American banking operations.

## 📄 Academic Paper

This repository contains the implementation of:

> **Herrera-Lara, L.A.; Herrera-Lara, R.C.** (2025). "A Machine Learning Framework for False Positive Detection in Application Performance Monitoring Systems: A Case Study in Latin American Banking". *Journal of Cybersecurity and Privacy*, MDPI.

**Key Results:**
- 🎯 **99.4% F1-score** in false positive detection
- 💰 **30,317% ROI** with $6.5M annual savings
- ⏱️ **545.7 hours/month** of analyst time recovered
- 🔍 **8 distinct FP patterns** identified and validated

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/[your-username]/elite-fp-detection.git
cd elite-fp-detection

# Install dependencies
pip install -r requirements.txt

# Run analysis
python elite_v6.py
```

## 📊 Features

### Core Capabilities
- **68 engineered features** across 5 categories
- **4-method ensemble** classification system
- **Cultural features** for Latin American banking (quincena, reconciliation)
- **SHAP-based explainability** for model transparency
- **Drift detection** for production monitoring
- **Comprehensive ROI** calculation

### Model Performance

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | 0.961 | 0.943 | 0.952 |
| Random Forest | 0.993 | 0.989 | 0.991 |
| **XGBoost** | **0.994** | **0.994** | **0.994** ✓ |
| LightGBM | 0.994 | 0.993 | 0.994 |

## 📁 Repository Structure
```
elite-fp-detection/
├── elite_v6.py                 # Main framework implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── data/
│   └── sample_dataset.csv      # Anonymized sample (1,000 records)
├── docs/
│   ├── METHODOLOGY.md          # Detailed methodology
│   ├── FEATURES.md             # Feature engineering guide
│   └── DEPLOYMENT.md           # Production deployment guide
└── examples/
    ├── basic_usage.ipynb       # Jupyter notebook tutorial
    └── custom_config.py        # Configuration examples
```

## 🔧 Configuration
```python
from elite_v6 import OptimizedConfigV6, EliteFalsePositiveAnalyzerV6

# Custom configuration
config = OptimizedConfigV6()
config.input_file = 'your_alerts.csv'
config.output_folder = 'results'
config.shap_enabled = True
config.drift_detection_enabled = True

# Initialize and run
analyzer = EliteFalsePositiveAnalyzerV6(config)
analyzer.load_data_optimized()
analyzer.engineer_features_optimized()
analyzer.classify_false_positives_advanced()
analyzer.build_models_with_temporal_cv()
analyzer.calculate_roi_comprehensive()
analyzer.generate_final_reports()
```

## 📖 Documentation

- **Paper**: [Full text](https://www.mdpi.com/journal/jcp) (pending publication)
- **Methodology**: [METHODOLOGY.md](docs/METHODOLOGY.md)
- **Feature Engineering**: [FEATURES.md](docs/FEATURES.md)
- **Deployment Guide**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 🎓 Citation

If you use this framework in your research, please cite:
```bibtex
@article{herrera2025elite,
  title={A Machine Learning Framework for False Positive Detection in 
         Application Performance Monitoring Systems: A Case Study in 
         Latin American Banking},
  author={Herrera-Lara, Luis Alberto and Herrera-Lara, Roberto Carlos},
  journal={Journal of Cybersecurity and Privacy},
  year={2025},
  publisher={MDPI}
}
```

## 👥 Authors

**Luis Alberto Herrera-Lara**  
Master's in Cybersecurity  
Universidad de Especialidades Espíritu Santo (UEES), Ecuador  
📧 luis.alberto.herrera.lara@gmail.com  
🔗 [ORCID: 0009-0000-8225-7536](https://orcid.org/0009-0000-8225-7536)

**Roberto Carlos Herrera-Lara**  
PhD Candidate in Telecommunications Engineering  
Universitat Politècnica de València (UPV), Spain  
📧 roberto.herrera.lara@gmail.com  
🔗 [ORCID: 0000-0003-0310-115X](https://orcid.org/0000-0003-0310-115X)

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🙏 Acknowledgments

- DevOps team at Diners Club Ecuador for data access
- Universidad de Especialidades Espíritu Santo (UEES) for academic support
- Universitat Politècnica de València (UPV) for research guidance
- Open-source community (scikit-learn, XGBoost, SHAP)

## 📞 Support

For questions or issues:
- 📧 Email: luis.alberto.herrera.lara@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/[your-username]/elite-fp-detection/issues)

---

**Note**: Due to confidentiality agreements, the complete dataset cannot be publicly shared. An anonymized sample dataset (1,000 records) is provided for testing and demonstration purposes.

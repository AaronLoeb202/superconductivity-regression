# Superconductivity Critical Temperature Prediction

This project aims to predict the **critical temperature (Tc)** at which materials become superconductors, based on their physical and chemical properties.

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/superconductivty+data), and contains more than 80 features describing the materials.

## 📌 Objectives

- Apply **linear regression** to scientific data.
- Preprocess the dataset using **normalization** and **log transformation**.
- Evaluate performance using **R² score** and **mean squared error (MSE)**.
- Identify and interpret the **most influential features**.
- Visualize results and draw physical insights from the model.

##  Tools & Technologies

- Python
- Jupyter Notebook / Spyder
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `ucimlrepo` (for loading the dataset)

## Results

- Baseline R² score: ~0.74  
- Log-transformed regression improved model robustness  
- Key features: weighted atomic radius, fusion heat, and valence entropy

## Visuals

The notebook includes:
- Histograms of the target variable
- True vs predicted scatter plots
- Feature importance charts

## How to Run

```bash
pip install -r requirements.txt
python superconductivity_regression.py

## Author

Aaron Loeb  
Machine Learning & Physics Enthusiast  
[LinkedIn](https://www.linkedin.com/)
#  Ordinal Classification via QWK-Optimized Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-orange?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Hackathon-20BEFF?style=for-the-badge&logo=kaggle)
![Optimization](https://img.shields.io/badge/Optimization-Nelder--Mead-success?style=for-the-badge)

A robust machine learning pipeline designed to solve an ordinal classification problem evaluated on the **Quadratic Weighted Kappa (QWK)** metric. Instead of standard classification, this pipeline uses regression to output continuous probabilities, which are then optimized into discrete bins using a custom Scipy-based threshold optimizer.

---

##  Key Features

*    QWK Threshold Optimization: Uses Nelder-Mead optimization to find the exact continuous thresholds that maximize Quadratic Weighted Kappa, improving significantly over standard rounding (0.5, 1.5, etc.).
*    Anomaly Extraction: Intelligently handles garbage string values (`17cm`, `500years`, `chotu`) scattered in numeric columns by extracting them into predictive metadata counts rather than just dropping them.
*    Dynamic Type Inference: Automatically scans the dataset, filtering out noise to correctly infer if a column should be treated as continuous numeric or Label-Encoded categorical.
*    ID Data Corruption Fixes: Bypasses classic Excel/CSV scientific notation corruption (`6.42E+11`) ensuring Kaggle submission evaluators don't fail.

---

##  Pipeline Architecture

1. **Data Ingestion** ➔ *Safe-loading IDs to prevent truncation.*
2. **Feature Engineering** ➔ *Counting anomalies and NaNs as new features.*
3. **Dynamic Type Parsing** ➔ *Separating pure numerics from disguised categoricals.*
4. **Model Training** ➔ *5-Fold Stratified K-Fold LightGBM (Regression).*
5. **Threshold Tuning** ➔ *Optimizing `[0.5, 1.5, 2.5]` bounds via `scipy.optimize`.*
6. **Submission Generation** ➔ *Formatting and patching corrupted submission IDs.*

---

##  Getting Started

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip install pandas numpy lightgbm scikit-learn scipy
Running the Code
If running on Kaggle:
Open a new notebook and attach the specific dataset (immihir/hackathon).
Paste the provided Jupyter Notebook code.
Run all cells. The submission will be generated in /kaggle/working/submission.csv.
If running Locally:
Update the train_path and test_path variables in Step 1 to point to your local .csv files.
Run the script via terminal or Jupyter:
code
Bash
python main.py
Results & Output
During training, the model outputs logs for 5-fold cross-validation. At the end of the script, the OptimizedRounder calculates the precise split thresholds.
Expected Output Log:
code
Text
Optimizing Thresholds for QWK...
Optimal Decision Thresholds found: [0.523, 1.481, 2.605]

Final Out-Of-Fold QWK Score: 0.99470
Success! Predictions saved to 'submission.csv'.
Developed for Machine Learning Kaggle Hackathon IIITK Challenge 2026

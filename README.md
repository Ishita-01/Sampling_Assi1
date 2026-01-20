
# Credit Card Fraud Detection: Sampling Technique Analysis

This project evaluates the performance of five different Machine Learning models across five distinct data sampling techniques. The primary objective is to determine which sampling method most effectively addresses the challenges of an imbalanced dataset in the context of credit card fraud detection.

## 1. Methodology

The project follows a rigorous data science pipeline:

### Data Preprocessing
The dataset used is `Creditcard_data (sampling).csv`. 
- **Challenge:** The original data was highly imbalanced (Class 0 >> Class 1).
- **Solution:** We applied **Manual Random Over-sampling** to the minority class (Fraud) until it matched the size of the majority class (Non-Fraud), resulting in a balanced dataset of 1,526 records.

### Sampling Techniques
Once the dataset was balanced, five different samples were extracted:
1. **Simple Random Sampling:** A 20% random fraction of the balanced population.
2. **Systematic Sampling:** Records selected at regular intervals (every 2nd record).
3. **Stratified Sampling:** A 60% sample that preserves the percentage of each class (Fraud/Not Fraud).
4. **Cluster Sampling:** Dividing the data into 5 groups (clusters) and selecting specific clusters for analysis.
5. **Bootstrap Sampling:** Random sampling performed with replacement, covering the full size of the balanced dataset.

### Machine Learning Models
Five classifiers were trained on each of the five samples:
- **M1:** Logistic Regression
- **M2:** Decision Tree
- **M3:** Random Forest
- **M4:** K-Nearest Neighbors (KNN, $k=1$)
- **M5:** Support Vector Machine (SVM)

---

## 2. Accuracy Matrix (Results)

The table below summarizes the accuracy (%) achieved by each model for every sampling technique.

| Model | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 (Logistic)** | 95.08 | 91.50 | 90.71 | 89.43 | 95.42 |
| **M2 (D-Tree)** | 98.36 | 99.35 | 98.91 | 98.37 | 99.35 |
| **M3 (R-Forest)** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| **M4 (KNN)** | 96.72 | 97.39 | 98.36 | 98.37 | 99.02 |
| **M5 (SVM)** | 81.97 | 64.05 | 63.93 | 66.67 | 73.20 |

---

## 3. Results Analysis & Visualizations

### Performance Comparison
Two main plots were generated to visualize the findings:
- **`performance_comparison.png`**: Compares the accuracy of each model (M1-M5) grouped by the sampling technique used.
- **`technique_effectiveness.png`**: Highlights the average accuracy of each sampling method across all tested models.

### Key Observations
- **Top Performer:** **Random Forest (M3)** demonstrated exceptional robustness, achieving 100% accuracy regardless of the sampling method used.
- **Effective Sampling:** **Bootstrap Sampling** and **Simple Random Sampling** yielded the most consistent high-performance results across all linear and non-linear models.
- **Model Sensitivity:** **SVM (M5)** showed the most significant drop in accuracy when moving from random sampling to systematic/stratified methods, indicating higher sensitivity to data ordering and distribution.

---

## 4. How to Run
1. Ensure the dataset `Creditcard_data (sampling).csv` is in the same directory as your script.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib

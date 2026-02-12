# **Maternal Hemoglobin Analysis - Linear Regression Model**
## Project Overview
     
     This project analyzes maternal hemoglobin levels at delivery using linear regression. The dataset contains 195 maternal patient records, including hemoglobin       results, Z-scores, gravidity (number of pregnancies), and testing method (LAMP vs SOC). 


          
## Learning Objectives

          This project serves as an educational example demonstrating:

     # Concept                                # What We Learned
     Linear Regression      ----------------- How to implement and interpret model coefficients
     R² Score               ----------------- Why 1.0 is not always good — context matters
     Residual Analysis      ----------------- Checking normality, homoscedasticity, independence
     Q-Q Plots              ----------------- Visual test for normality of residuals
     Outlier Detection      ----------------- Z-score method and box plot visualization
     Data Leakage           ----------------- Why you shouldn't predict a transformed target
     GitHub Documentation   ---------------- How to clearly communicate statistical findings




## Dataset Description
     
      Column                                 Description                             Range
      Questionnaire_Number ----------------- Unique patient ID (GGH-001, etc.)         — 
      Event_Name           ----------------- Testing method: LAMP or SOC              2 categories 
      Result               ----------------- Hemoglobin level (g/dL)                  7.3 - 14.2 
      Gravidity            ----------------- Number of pregnancies                    1 - 5 
      Z-Score              ----------------- Standardized hemoglobin                 -3.17 - 2.04 




## Sample Data:
     
     Questionnaire       Event    Result     Gravidity    Z-Score 
     
      GGH-001 --------- LAMP      12.3       1             0.606 
      GGH-002 --------- SOC       11.4       2            -0.074 
      GGH-004 --------- LAMP      12.0       1             0.380 
      GGH-005 --------- LAMP      10.4       1            -0.830 
      GGH-006 --------- SOC       13.5       1             1.514 





## Analysis Steps

     Load Data      ◄── Excel file with hemoglobin measurements

         ↓

     Explore Data    ◄── Head (), describe (), pairplot()

         ↓
     Train/Test     ◄── 70% train, 30% test
     Split          

         ↓

    Linear Model     ◄── Z-Score = β·Result + α

         ↓

     Evaluate       ◄── R², MAE, MSE, RMSE

         ↓

     Residual        ◄── Distribution, Q-Q plot
     Analysis       

         ↓
 
     Outlier        ◄── Z-score threshold = 3
    Detection      







## Results & Interpretation
     
      Model Performance Metrics
     
     | Metric         | Value          | Interpretation |
     | R² Score       | 1.0            | Perfect score — explains 100% of variance |
     | Coefficient    | 0.756          | Each +1 g/dL Hb → +0.756 Z-Score |
     | Intercept      | -8.693         | Z-Score = 0 when Hb = 11.5 g/dL |
     | MAE            | 7.32e-16       | Zero prediction error |
     | RMSE           | 8.63e-16       | Zero prediction error |



      Clinical Insights (From the Data Itself)
     Despite the modeling issue, the raw data reveals useful information:
     
     | Finding           | Value              | Clinical Meaning |
     | Mean Hemoglobin   | 11.5 g/dL          | Normal for pregnancy (physiological dilution) |
     | Hemoglobin Range  | 7.3 - 14.2 g/dL    | From severe anemia to normal |
     | Outlier           | 7.3 g/dL           | 1 patient with severe anemia — needs follow-up |
     | Gravidity Effect  | Weak positive      | Slightly higher Hb in multiparous women  |
     | Testing Methods   | LAMP & SOC         | Both point-of-care and standard methods present |






## Visualizations

      | Plot                   | File                      | What It Shows |
      | Pair Plot              | pairplot.png              | Relationships between all variables  |
      | Predicted vs Actual    | predicted_vs_actual.png   | Perfect diagonal line — predictions = actual values |
      | Residual Density       | residual_density.png      | Bell-shaped curve — residuals normally distributed |
      | Q-Q Plot               | qq_plot.png               | Points on straight line — confirms normality |
      | Box Plot               | boxplot.png               | Outlier at 7.3 g/dL — severe anemia case |



![image alt](https://github.com/danblacknote/Regression-Models-Using-Python/blob/a3900f3de6b1516eacc08c88ba27a8cd66b90faf/Analysis_Graphs/Probability%20Plot.png)
![image alt](https://github.com/danblacknote/Regression-Models-Using-Python/blob/a3900f3de6b1516eacc08c88ba27a8cd66b90faf/Analysis_Graphs/Z-Score%20and%20Y-Pridict%20Graph%202.png)
![image alt](https://github.com/danblacknote/Regression-Models-Using-Python/blob/a3900f3de6b1516eacc08c88ba27a8cd66b90faf/Analysis_Graphs/Density%20Plote.png)






## Statistical Concepts Implemented:
       Linear Regression Implementation — sklearn workflow
       Model Evaluation — R², MAE, MSE, RMSE
       Residual Analysis — Normality, homoscedasticity
       Q-Q Plots — Visual normality testing
       Outlier Detection — Z-score method (threshold = 3)
       Data Visualization — Seaborn, Matplotlib
       GitHub Documentation — Clear communication of results







##  Repository Structure
              
            📁 Regression-Model-Using-Python/
            |
            |
            ├── 📁 Result_Graphes/
            │   ├── pairplot.png
            │   ├── predicted_vs_actual.png
            │   ├── residual_density.png
            │   ├── qq_plot.png
            │   └── boxplot.png
            |
            |
            └── 📁 Analysis_Results/
            |       └── Finding_interpretation.md
            |
            |
            |
            ├── 📁 DataSet/
            │    └── Maternal_Hemoglobin_at_Delivery.xlsx
            |
            │
            |
            ├── 📄 README.md                  You are here
            ├── 📄 linear_regression.py       Main analysis script



## Summary

      feat: Complete linear regression analysis on maternal hemoglobin data
      
       Model Performance:
      - R² = 1.0, MAE ≈ 0, RMSE ≈ 0 (mathematically perfect)
      - Coefficient: 0.756, Intercept: -8.693
      
      CRITICAL NOTE:
      Perfect fit occurs because Z-Score is mathematically derived from the result:
      Z = (Result - μ)/σ. The model simply reverse-engineered this formula.
      **Findings:**
      - Population mean Hb: 11.5 g/dL (normal for pregnancy)
      - One outlier detected: 7.3 g/dL (severe anemia — follow-up required)
      - Weak positive correlation between gravidity and hemoglobin
      - All regression assumptions satisfied


## Author
      Data Manager/Data Analyst / Statistician  
      Deneke Zewdu 
      Date: - Feb 12,2026

## License
      This project is licensed under the MIT License and can be used for educational purposes only. Not for clinical use.



# RNA-structure-prediction
In this project we aim to predict the three-dimensional structure of RNA molecules using machine learning approaches.

# Introduction
This project is based on RNA, a molecule that plays a vital role in biological systems, coming
from DNA, is a direct transcript of our genetic information and allows protein synthesis in our
cells. These proteins are fundamental for proper body function, making RNA an essential
molecule for survival. When RNA molecules are synthesized, they are simple, linear chains
without many functions, but quickly fold into complex tridimensional structures and acquire
specific functions within the cell. Understanding RNA structure is crucial for understanding
how biological systems work.
In this project we aim to predict the three-dimensional structure of RNA molecules using
machine learning approaches. By analyzing the DNA sequences that the RNA comes from
we can predict the final structure of the RNA molecule, and therefore, its function with further
analysis. This has a high importance for a variety of biological processes, including gene
regulation or protein synthesis mechanisms. Prediction of RNA is increasing its importance
every day in the molecular biology and bioinformatics area and the advancements in
computational methods and machine learning makes this task more feasible and accurate.

# Materials and Methods
To predict RNA structures, we used a database consisting of a big amount of DNA
sequences, from these sequences we can predict the RNA sequence and structure.
Different machine learning models were used for prediction: SVM, Random Forest and
XGBoost, being the best and the final one, Random Forest. We measured the values of:
Sensitivity, PPV, MCC and F1-score. These models are based on traditional machine
learning approaches but we also worked with Eternafold, a specific machine learning model
for predicting RNA molecule structures, these types of models are pretrained models used
for RNA and other molecules.

Key Steps:
● Sequence Input: DNA sequences obtained from Kaggle.
● Exploratory Data analysis
● Prediction Algorithm: The computational tools Random Forest and Eternafold.

Used Parameters
The parameters used during the prediction process were:
1. Sensitivity: measures proportion of TP and FP + FN.
2. PPV: Positive Predictive Value measures proportion of TP out of all positive
predictions made by the model.
3. Matthews Correlation Coefficient MCC: takes into account TP, TN, FP AND FN.
4. F-measure (F1-score): is the harmonic mean of precision and recall, in RNA
structure prediction it helps evaluate the overall performance of the model.

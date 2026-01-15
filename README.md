# Traffic Accident Prediction Using SUMO and Machine Learning

## Project Overview

This project presents a simulation-driven framework for predicting rear-end traffic collisions using traffic flow insights and machine learning techniques. Due to the lack of large-scale real-world traffic accident datasets, the system leverages SUMO (Simulation of Urban MObility) to generate realistic traffic scenarios and sensor-based data for accident prediction.

The framework focuses on time-series traffic data analysis and evaluates both traditional machine learning models and deep learning approaches to accurately identify accident-prone situations in highly imbalanced datasets.

## Objectives

- Simulate realistic traffic environments and rear-end collision scenarios using SUMO

- Extract and preprocess sensor-level traffic flow features

- Apply machine learning and deep learning models for accident prediction

- Handle severe class imbalance using appropriate sampling techniques

- Evaluate models using precision, recall, and F1-score instead of accuracy

## Technologies & Tools Used

- **Traffic Simulator:** SUMO

- **Programming Language:** Python

- **Machine Learning:**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine(SVM)
  - K-Nearest Neighbors(KNN)
  - Naive Bayes

- **Deep Learning:** Long Short-Term Memory(LSTM)

- **Libraries & Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
  - Matplotlib / Seaborn

## Dataset Description

Traffic data generated using SUMO simulations

- **Sensor-based features include:**

  - Vehicle flow

  - Vehicle occupancy

  - Average speed

  - Number of vehicles entered

  - Timestamp

Dataset is highly imbalanced, with accident events representing ~1% of total data

Synthetic Minority Oversampling Technique (SMOTE) used to address imbalance

## Methodology

1. Simulate traffic and rear-end collision scenarios in SUMO
2. Collect sensor data and convert XML outputs to CSV format
3. Preprocess data and generate labeled datasets
4. Apply time-shifted features to capture temporal dependencies
5. Train and evaluate ML and LSTM models
6. Compare performance using precision, recall, and F1-score

## Results & Key Findings

- LSTM outperformed traditional ML models for time-series traffic data

- Achieved up to 91% F1-score for accident prediction at optimal time shifts

- Demonstrated the importance of temporal dependencies in traffic accident prediction

- F1-score proved more reliable than accuracy for imbalanced accident datasets

## Repository Structure
```
├── data/
│   └── Main.csv
├── notebooks/
│   └── ML_Traffic2.ipynb
├── simulation/
│   └── SUMO configuration files
├── results/
│   └── graphs and evaluation metrics
├── README.md

```

 ## How to Run the Project

1. Clone the repository
```
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Jupyter Notebook

```
jupyter notebook
```

4. Open ML_Traffic2.ipynb and execute cells sequentially

## Research Paper

Title: A Data-Driven Approach to Predicting Rear-End Collisions Through Traffic Insights
This project is based on an academic research study focusing on intelligent transportation systems and accident prediction using simulated traffic data.

## Future Enhancements

- Integrate real-time traffic data from sensors or APIs

- Extend framework to detect multiple types of accidents

- Deploy prediction model as a real-time ITS service

- Improve scalability with larger and diverse road networks



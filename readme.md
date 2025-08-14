# Metaheuristic Optimization for Logistic Regression:
Optimize logistic regression hyperparameters using Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and a Hybrid ACO + GridSearchCV approach to improve classification performance.

Dataset: Breast Cancer Wisconsin (Diagnosis)
Goal: Achieve optimal accuracy by tuning the regularization parameter (C) of Logistic Regression.

# Featured and Presented at Symposium 2024 (Saint Peter's University)
![Presentation 1.jpeg](Presentation%2FPresentation%201.jpeg)
![Symposium May 2024.jpeg](Presentation%2FSymposium%20May%202024.jpeg)
![Certificate.jpeg](Presentation%2FCertificate.jpeg)

# Features
1. Clean and scalable codebase using modular Python files

2. Implements PSO, ACO, and a Hybrid metaheuristic optimization

3. Visualizes accuracy vs. execution time

4. Tracks confusion matrix for each algorithm

5. Designed for performance comparison and research use

6. Lightweight and reproducible
 
# How it Works
![project.png](project.png)
In nature, optimization is key to survival. Ants discover food by laying down pheromone trails—this inspires Ant Colony Optimization (ACO),
where better paths are reinforced over time. Birds fly in flocks,
constantly adjusting based on their own and their neighbors’ positions—this forms the idea behind Particle Swarm Optimization (PSO),
where solutions evolve through shared knowledge. 
But what happens when global search needs local precision? 
That’s where the Hybrid Optimizer comes in. 
It combines ACO's broad search ability with GridSearchCV's fine-tuning skills to optimize logistic regression.
The result is smarter, more accurate models, guided by the same principles that power nature.

# Project Structure:
Metaheuristic_Optimization_for_Logistic_Regression/
│
├── .gitignore
├── README.md
├── requirements.txt
├── project.png
│
├── data/
│   ├── data.csv                  # Original datasets
│
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_data_preprocessing_and_feature_scaling.ipynb
│   ├── 03_baseline_logistic_regression_model.ipynb
│   ├── 04_hyperparameter_tuning_with_gridsearchcv.ipynb
│   ├── 05_metaheuristic_optimization_pso.ipynb
│   ├── 06_metaheuristic_optimization_aco.ipynb
│   ├── 07_hybrid_optimization_approach.ipynb
│   └── 08_final_comparison_and_insights.ipynb
│
├── results/
│   ├── model_performance.csv
│   ├── images/
│   │   ├── accuracy_vs_time.png
│   │   ├── ACO_confusion_matrix.png
│   │   ├── Hybrid_confusion_matrix.png
│   │   └── PSO_confusion_matrix.png
│
├── scripts/
│   ├── generate_results.py      # execution script produces results                   
│   ├── aco_optimizer.py
│   ├── pso_optimizer.py
│   └── hybrid_optimizer.py
│
├── tests/
     ├── test_aco.py
     ├── test_hybrid.py
     └── test_pso.py

# Installation:
Clone the repo and install dependencies:
git clone https://github.com/yourusername/Metaheuristic_Optimization_for_Logistic_Regression.git
cd Metaheuristic_Optimization_for_Logistic_Regression
pip install -r requirements.txt

# ▶️ How to Run:
Run the main script to execute all three optimizers and visualize results:
python generate_results.py
All output will be saved in the results/ folder, including:
1. Accuracy scores

2. Best C values

3. Confusion matrices

4. Execution time charts

# Algorithms Used:
1. PSO: Particles adjust based on personal & global best scores

2. ACO: Simulates ant behavior via pheromone updates

3. Hybrid: ACO for exploration → GridSearch for fine-tuning

# Results:
Optimizer	Best_C		Accuracy
PSO		    6.279		0.9825
ACO		    33.34		0.9825
Hybrid		9.12		0.9825
The hybrid optimizer provides consistent and stable results by refining ACO's exploration via grid search.

# 📦 Dependencies
1. Python 3.7+
2. numpy
3. pandas
4. scikit-learn
5. matplotlib
6. seaborn

# Install all with:
pip install -r requirements.txt

# Testing
Sample unit test for ACO optimizer:
cd tests
python test_aco_optimizer.py

# Contributing
Pull requests are welcome! For major changes, please open an issue first.

# 📧 Contact
Created with ❤️ by Ajay Raj Singh
Feel free to connect on LinkedIn or open issues for suggestions!
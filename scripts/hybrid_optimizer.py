import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def run_hybrid_optimizer(X_train, y_train, X_test, y_test):
    # ACO Parameters
    num_ants = 10
    num_iterations = 20
    lb, ub = 0.01, 100
    evaporation_rate = 0.5
    pheromone = np.ones(10)
    discretized_C = np.linspace(lb, ub, 10)
    best_score = 0
    best_C = None

    def fitness_function(c_val):
        model = LogisticRegression(C=c_val, max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    start_time_aco = time.time()

    for iteration in range(num_iterations):
        ant_solutions = []
        ant_scores = []

        for _ in range(num_ants):
            probs = pheromone / pheromone.sum()
            idx = np.random.choice(range(len(discretized_C)), p=probs)
            C_value = discretized_C[idx]

            score = fitness_function(C_value)
            ant_solutions.append(idx)
            ant_scores.append(score)

        pheromone = (1 - evaporation_rate) * pheromone
        for idx, score in zip(ant_solutions, ant_scores):
            pheromone[idx] += score

        max_idx = np.argmax(ant_scores)
        if ant_scores[max_idx] > best_score:
            best_score = ant_scores[max_idx]
            best_C = discretized_C[ant_solutions[max_idx]]

    end_time_aco = time.time()
    aco_time = end_time_aco - start_time_aco

    # GridSearch around ACO best C
    param_grid = {'C': [round(best_C + delta, 2) for delta in [-2, -1, 0, 1, 2] if best_C + delta > 0]}
    lr_model = LogisticRegression(max_iter=1000, solver='liblinear')

    start_time_grid = time.time()
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    end_time_grid = time.time()

    grid_time = end_time_grid - start_time_grid
    total_time = aco_time + grid_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "best_c": grid_search.best_params_['C'],
        "accuracy": test_acc,
        "execution_time": total_time,
        "model": best_model
    }

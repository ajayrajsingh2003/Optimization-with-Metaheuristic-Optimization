from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import time

def aco_optimizer(X_train, y_train, X_test, y_test):
    def fitness(C_value, X_train, y_train, X_test, y_test):
        model = LogisticRegression(C=C_value, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    # ACO Parameters
    num_ants = 10
    num_iterations = 20
    lb, ub = 0.01, 100
    evaporation_rate = 0.5
    pheromone = np.ones(10)  # initial pheromone levels for discretized choices
    discretized_C = np.linspace(lb, ub, 10)  # discretized search space
    best_score = 0
    best_C = None

    start_time = time.time()

    for iteration in range(num_iterations):
        ant_solutions = []
        ant_scores = []

        for _ in range(num_ants):
            probs = pheromone / pheromone.sum()
            idx = np.random.choice(range(len(discretized_C)), p=probs)
            C_value = discretized_C[idx]

            score = fitness(C_value, X_train, y_train, X_test, y_test)
            ant_solutions.append(idx)
            ant_scores.append(score)

        pheromone = (1 - evaporation_rate) * pheromone
        for idx, score in zip(ant_solutions, ant_scores):
            pheromone[idx] += score

        max_idx = np.argmax(ant_scores)
        if ant_scores[max_idx] > best_score:
            best_score = ant_scores[max_idx]
            best_C = discretized_C[ant_solutions[max_idx]]

    end_time = time.time()
    execution_time = end_time - start_time

    best_model = LogisticRegression(C=best_C, max_iter=1000)
    best_model.fit(X_train, y_train)

    return {
        "model": best_model,
        "best_c": best_C,
        "accuracy": best_score,
        "execution_time": execution_time
    }

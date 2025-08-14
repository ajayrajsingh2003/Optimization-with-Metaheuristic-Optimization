import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def pso_optimizer(X_train, y_train, X_test, y_test):
    def fitness_function(c):
        try:
            model = LogisticRegression(C=c, max_iter=1000, solver='liblinear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
        except Exception:
            return 0.0  # Return lowest score on error

    # PSO Parameters
    num_particles = 10
    num_iterations = 20
    w = 0.5   # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient

    lb, ub = 0.01, 100  # Boundaries for C
    particles = np.random.uniform(lb, ub, num_particles)
    velocities = np.zeros(num_particles)
    pbest_positions = particles.copy()
    pbest_scores = np.array([fitness_function(c) for c in particles])
    gbest_position = pbest_positions[np.argmax(pbest_scores)]
    gbest_score = max(pbest_scores)

    start_time = time.time()

    for _ in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)

            score = fitness_function(particles[i])
            if score > pbest_scores[i]:
                pbest_positions[i] = particles[i]
                pbest_scores[i] = score

        best_idx = np.argmax(pbest_scores)
        if pbest_scores[best_idx] > gbest_score:
            gbest_score = pbest_scores[best_idx]
            gbest_position = pbest_positions[best_idx]

    end_time = time.time()
    execution_time = end_time - start_time

    final_model = LogisticRegression(C=gbest_position, max_iter=1000, solver='liblinear')
    final_model.fit(X_train, y_train)

    return {
        "best_c": gbest_position,
        "accuracy": gbest_score,
        "execution_time": execution_time,
        "model": final_model
    }

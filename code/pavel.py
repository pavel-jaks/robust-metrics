import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


class WassersteinApproximation():
    def __init__(self, regularization: float = 1e3, stopping_criterion:float = 1e-5):
        self.regularization = regularization
        self.stopping_criterion = stopping_criterion

    def compute_vectors_distance(self, x, y, cost_matrix):
        if any(x == 0):
            raise Exception('Wrong input')
        
        # Algorithm from paper https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
        u_vector = np.ones(len(x)) / len(x)
        K_matrix = np.exp(- self.regularization * cost_matrix)
        K_tilde_matrices = np.diag(1 / x) @ K_matrix

        for _ in range(500):
            u_vector = 1 / (K_tilde_matrices @ (y / (K_matrix.T @ u_vector)))
        
        v_vector = y / (K_matrix.T @ u_vector)
        return np.diag(u_vector) @ K_matrix @ np.diag(v_vector)


# Generate some data (they need to be probability distributions)
n = 6
x = np.random.rand(n)
y = np.random.rand(n)
x /= np.sum(x)
y /= np.sum(y)

# Regularization parameter
regularization = 0.3

# Cost matrix
cost_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        cost_matrix[i,j] = abs(i-j)

# Solution via the iterative method
metric = WassersteinApproximation(regularization=regularization)
solution1 = metric.compute_vectors_distance(x,y,cost_matrix)
# Other solution will be based on solving an optimization problem

# We need some initial solution
n_sq = n*n
x0 = np.ones(n_sq)
x0 /= np.sum(x0)

# The objective
def objective(p):
    p_nonzero = p[p>0]
    kl = -np.sum(p_nonzero*np.log(p_nonzero))
    p = p.reshape(n,n)
    return np.sum(p * cost_matrix) - 1/regularization*kl

# We need p to be probability
n_sq = n*n
bounds = Bounds(np.zeros(n_sq), np.ones(n_sq))
linear_constraint = LinearConstraint(np.ones((1,n_sq)), [1], [1])

# We need linear constraints. con1 says p*1 = x, con2 says p'*1 = y
rm1 = np.kron(np.eye(n), np.ones(n))
rm2 = np.kron(np.ones(n), np.eye(n))
con1 = LinearConstraint(rm1, x, x)
con2 = LinearConstraint(rm2, y, y)

# Verify that the constraints make sense. Both numbers are small
np.linalg.norm(rm1 @ solution1.reshape(n_sq) - x)
np.linalg.norm(rm2 @ solution1.reshape(n_sq) - y)

# Get the solution by solving the optimalization problem
res = minimize(objective, x0, method='trust-constr',
               constraints=[linear_constraint, con1, con2],
               options={'verbose': 1}, bounds=bounds)
solution2 = res.x

# Verify that the solutions are identical
np.linalg.norm(solution1.reshape(n_sq) - solution2)

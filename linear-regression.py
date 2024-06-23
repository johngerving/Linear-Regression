import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self):
        self.coef = None

    '''
    function gaussianElimination:
        arguments:
            matrix - a matrix where n = m + 1
        return:
            A - a matrix in reduced row echelon form, where the last column contains the solution to the system of equations
    '''
    def gaussianElimination(self, matrix):
        A = np.copy(matrix).astype(np.float64)

        # m is number of rows, n is number of columns
        m, n = A.shape

        if m + 1 != n:
            raise Exception(f"Matrix with shape ({m}, {n}) is not a square system.")

        h = 0 # Pivot row
        k = 0 # Pivot column

        while h < m and k < n:
            # Find the kth pivot column
            i_max = np.argmax(abs(A[h:, k])) + h
            if A[i_max, k] == 0:
                # No pivot in this column, pass to next column
                k += 1
            else:
                # Swap row h and row i_max
                A[[h, i_max]] = A[[i_max, h]]
                # For all rows except pivot:
                for i in range(m):
                    if i != h:
                        multiplier = A[i, k] / A[h, k]
                        # Fill other parts of pivot columns with zeros
                        A[i, k] = 0
                        # For all remaining elements in current row:
                        for j in range(k+1, n):
                            A[i, j] = A[i, j] - A[h, j] * multiplier

                # Increment pivot row and column
                h += 1
                k += 1

        # Solve the system by dividing the last element of each row by the coefficient in that row
        i = 0

        while i < m and i < n-1:
            A[i, -1] /= A[i, i]
            A[i, i] = 1

            i += 1
        
        return A
    
    def fit(self, X, y):
        # Convert X and y to Numpy arrays if necessary
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy(copy=True).astype(np.float64)
        else:
            X = np.copy(X).astype(np.float64)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy(copy=True).astype(np.float64)
        else:
            y = np.copy(y).astype(np.float64)

        X_shape = X.shape
        y_shape = y.shape
        
        if X_shape[0] != y_shape[0]:
            raise Exception('Number of instances of independent and dependent variables must match')
        if y.ndim != 1:
            raise Exception('Dependent variable must be 1-dimensional')

        # Reshape X into columns if there is only one regressor
        if(X.ndim == 1):
            X = np.reshape(X, (X_shape[0], 1))
        y = np.reshape(y, (y_shape[0], 1))

        X = X[~np.isnan(X).any(axis=1)]
        y = y[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(y).any(axis=1)]
        y = y[~np.isnan(y).any(axis=1)]

        num_regressors = X.shape[1] # Number of independent variables
        num_instances = X.shape[0] # Number of data points

        # Create matrix to solve for coefficients
        num_matrix_rows = num_regressors + 1
        num_matrix_cols = num_regressors + 2

        matrix = np.zeros(shape=(num_matrix_rows, num_matrix_cols))
        for i in range(num_matrix_rows):
            for j in range(num_matrix_cols - 1):
                if i == 0 and j == 0: # Upper left corner of matrix
                    matrix[i, j] = num_instances
                elif i == 0 and j > 0: # First row of matrix
                    matrix[i, j] = np.sum(X[:, j - 1])
                elif j == 0 and i > 0: # First column of matrix
                    matrix[i, j] = np.sum(X[:, i - 1])
                else: # Everything else
                    matrix[i, j] = np.sum(X[:, i - 1] * X[:, j - 1])
        
        # Populate last column of matrix representing output of linear equations
        matrix[0, -1] = np.sum(y)
        for i in range(1, num_matrix_rows):
            matrix[i, -1] = np.sum(X[:, i - 1] * y[:, 0])

        # Solve system for coefficients
        self.coeff = self.gaussianElimination(matrix)[:, -1].flatten()

linreg = LinearRegression()

data = pd.read_csv("train.csv")

# linreg.fit(data['x'], data['y'])

X = np.array([[18, 52],
              [24, 40],
              [12, 40],
              [30, 48],
              [30, 32],
              [22, 16]])
y = np.array([144, 142, 124, 64, 96, 92])

linreg.fit(X, y)
print(linreg.coeff)
import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self):
        self.coeff = None

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
    
    '''
    function fit:
        arguments:
            X - input array or dataframe
            y - output array
        post:
            Fills self.coeff with coefficients for linear regression
    '''
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

    '''
    function predict:
        arguments: 
            X - number, Numpy array, or Pandas dataframe; if array or dataframe, each row contains an instance to be predicted
        return:
            output - an array containing an output for each instance in X
    '''
    def predict(self, X):
        if self.coeff is None:
            raise Exception("Model has not been fit")
        
        # If only one regressor, allow single value to be passed in
        if isinstance(X, (int, float, complex)) and not isinstance(X, bool) and self.coeff.size == 2:
            return self.coeff[0] + self.coeff[1] * X
        elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy(copy=True).astype(np.float64)

        X_shape = X.shape
        if(X.ndim == 1):
            X = np.reshape(X, (X_shape[0], 1))

        # If shape of input and coefficients do not match
        if X.shape[1] != self.coeff.size - 1:
            raise Exception("Shape of input must match number of coefficients")
        
        # For each regressor, multiply by the input variable and add to output
        output = np.zeros(shape=(X.shape[0],))
        for i in range(output.size):
            output[i] += self.coeff[0]
            for coeff in range(1, self.coeff.size):
                output[i] += self.coeff[coeff] * X[i, coeff-1]
                
        return output
    
    def squaredResiduals(self, y_pred, y_actual):
        if self.coeff is None:
            raise Exception("Model has not been fit")
        if(len(y_pred) != len(y_actual)):
            raise Exception("Number of instances of predicted and actual do not match")
        if isinstance(y_actual, pd.Series):
            y_actual = y_actual.to_numpy(copy=True).astype(np.float64)
        y_pred = np.copy(y_pred)

        # Remove NaNs
        y_pred = y_pred[~np.isnan(y_pred)]
        y_actual = y_actual[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_actual)]
        y_actual = y_actual[~np.isnan(y_actual)]

        # Return sum of squares of difference between actual value and predicted value
        squaredResiduals = np.sum(np.square(y_actual - y_pred))

        return squaredResiduals

    def totalSumOfSquares(self, y):
        if self.coeff is None:
            raise Exception("Model has not been fit")
        if isinstance(y, pd.Series):
            y = y.to_numpy(copy=True).astype(np.float64)

        # Remove NaNs
        y = y[~np.isnan(y)]

        mean = np.mean(y)

        # Return sum of squares of difference between value and mean value
        totalSumOfSquares = np.sum(np.square(y - mean))
        return totalSumOfSquares
    
    '''
    R^2 represents the variation in the data that can be explained by the relationship between the independent and dependent variables.
    '''
    def r_squared(self, y_pred, y_actual):
        # Return ratio between sum of squared residuals and sum of squares subtracted from 1
        return 1 - self.squaredResiduals(y_pred, y_actual) / self.totalSumOfSquares(y_actual)
    
    '''
    Root Mean Squared Error
    '''
    def RMSE(self, y_pred, y_actual):
        if self.coeff is None:
            raise Exception("Model has not been fit")
        if(len(y_pred) != len(y_actual)):
            raise Exception("Number of instances of predicted and actual do not match")
        if isinstance(y_actual, pd.Series):
            y_actual = y_actual.to_numpy(copy=True).astype(np.float64)
        y_pred = np.copy(y_pred)

        # Remove NaNs
        y_pred = y_pred[~np.isnan(y_pred)]
        y_actual = y_actual[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_actual)]
        y_actual = y_actual[~np.isnan(y_actual)]

        n = y_actual.size

        # Calculate sum of squared residuals and divide by number of data points, then take square root
        sumSquaredResiduals = self.squaredResiduals(y_pred, y_actual)
        meanSquaredError = sumSquaredResiduals / n
        rootMeanSquaredError = np.sqrt(meanSquaredError)

        return rootMeanSquaredError


linreg = LinearRegression()

data = pd.read_csv("train.csv")

data = data.dropna()

X = data['x']
y = data['y']

linreg.fit(X, y)
y_pred = linreg.predict(X)
print("R^2:", linreg.r_squared(y_pred, y))
print("RMSE:", linreg.RMSE(y_pred, y))
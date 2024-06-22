import numpy as np

class LinearRegression():

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

linreg = LinearRegression()

matrix = np.array([[6, 136, 228, 662],
                   [136, 3328, 5128, 14312],
                   [228, 5128, 9488, 25744]])

newMatrix = linreg.gaussianElimination(matrix)
print(newMatrix)
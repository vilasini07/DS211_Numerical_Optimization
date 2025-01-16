# Date: 16/01/2025
import numpy as np
import pandas as pd

# use pandas to load real_estate_dataset.csv
df=pd.read_csv('real_estate_dataset.csv')

# get the number of samples and features
n_samples, n_features = df.shape
print(f"Number of samples, features: {n_samples, n_features}")

# get the names of columns
columns = df.columns

#save the column names to a file for accessing later as text file
np.savetxt('columns.txt', columns, fmt='%s')

#Use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as feature
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values

# Use price as target
y = df['Price'].values

print(f"Shape of X: {X.shape}\n")
print(f"Data type of X: {type(X)}\n")

# Get the number of samples and features in X
n_samples, n_features = X.shape

# Build a linear model to predict price from the four features in X
# Make an array of coefficients of the size of n_features+1, initialized to 1.

coefs=np.ones(n_features+1)

# Here we are adding the bias manually for each data point:
# Predict the price for each sample in X
predictions_bydefn = np.dot(X, coefs[1:]) + coefs[0]

# FINDING THE PREDICTIONS USING INITIAL GUESS OF COEFFICIENTS
# Here we are adding the bias using matrix operations(including an extra column in X so that bias gets added in dot product):
# Append a column of ones to X
X = np.hstack((np.ones((n_samples,1)), X))

# Predict the price for each sample in X
predictions = np.dot(X, coefs)

# See if all entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)

print(f"Are the predictions same with X*coefs[1:]+coefs[0] and X*coefs: {is_same}\n")

# HERE WE FIND ERRORS USING A LOOP AND MATRIX OPERATIONS
# Calculate the error using predictions and y
errors = y-predictions

# Calculate the relative error
rel_errors = errors/y

# Calculate the mean of square of errors using a loop
loss_loop=0
for i in range(n_samples):
    loss_loop += errors[i]**2
loss_loop /= n_samples

# Calculate the mean of square of errors using matrix operations    
loss_matrix=np.transpose(errors)@errors/n_samples

# Compare the two methods of calculating the mean of square of errors
is_diff = np.allclose(loss_loop, loss_matrix)
print(f"Are the two methods of calculating the mean of square of errors same: {is_diff}\n")

# Print the size of errors and its L2 norm
print(f"Size of errors: {errors.shape}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}\n")
print(f"L2 norm of relative errors: {np.linalg.norm(rel_errors)}\n")

# What is my optimization problem?
# I want to find the coefficients that minimize the mean of square of errors
# This problem is called least squares problem

# Objective function: f(coefs) =1/n_samples* \sum_{i=1}^{n_samples} (y_i - \sum_{j=1}^{n_features+1} X_{ij} coefs_j)^2
# The solution is the point in R^(n_samples+1) at which gradient of the objective function is zero

# Write the loss matrix in terms of the data and the coefs
loss_matrix = np.transpose(y-np.dot(X, coefs))@(y-np.dot(X, coefs))/n_samples

# Calculate the gradient of the loss function with respect to coefs
grad_matrix = -2/n_samples*np.transpose(X)@(y-np.dot(X, coefs))

# We set grad_matrix to zero and solve for coefs
# SOLVING NORMAL EQUATION DIRECTLY BY FINDING THE INVERSE OF X^T @ X:
# X^T @ y = X^T @ X @ coefs (known as normal equation)
# coefs = (X^T @ X)^{-1} @ X^T @ y
coefs = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y

# Save coefs to a file for viewing
np.savetxt('coefs.csv', coefs, delimiter=',')

# Calculate the predictions using optimal coefs
predictions_model = X@coefs

# Calculate the errors using optimal coefs
errors_model = y-predictions_model

# Print the L2 norm of errors_model
print(f"L2 norm of errors using optimal coefs: {np.linalg.norm(errors_model)}\n")

# print the L2 norm of relative errors using optimal coefs
rel_errors_model = errors_model/y
print(f"L2 norm of relative errors using optimal coefs: {np.linalg.norm(rel_errors_model)}\n")

# WE NOW USE ALL THE FEATURES IN THE DATASET TO BUILD A LINEAR MODEL TO PREDICT PRICE
X =df.drop('Price', axis=1).values
y = df['Price'].values

# Get the number of samples and features in X
n_samples, n_features = X.shape
print(f"Number of samples, features: {n_samples, n_features}\n")

# SOLVING NORMAL EQUATIONS USING ALL FEATURES
# Solve the linear model using normal equation
X=np.hstack((np.ones((n_samples,1)), X))
coefs = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y

# Calculate the predictions using coefs
predictions_all = np.dot(X, coefs)

# Calculate the errors using coefs
errors_all = y-predictions_all
rel_errors_all = errors_all/y
print(f"L2 norm of errors using all features: {np.linalg.norm(errors_all)}\n")
print(f"L2 norm of relative errors using all features: {np.linalg.norm(rel_errors_all)}\n")

# Save the coefs to a file named coefs_all.csv
np.savetxt('coefs_all.csv', coefs, delimiter=',')

# Calculate the rank of X^T @ X
rank_XTX = np.linalg.matrix_rank(np.transpose(X)@X)
print(f"Rank of X^T @ X: {rank_XTX}\n")

# Solve the normal equation using matrix decomposition
# QR FACTORIZATION
Q, R=np.linalg.qr(X)

print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")

# X=QR
# X^T X = R^T Q^T Q R = R^T R
# X^T y = R^T Q^T y
# R*coefs = Q^T y

b=np.dot(np.transpose(Q), y)
# Loop for solving R*coefs = Q^T y using back substitution
coefs_QR=np.zeros(n_features+1)

for i in range(n_features, -1, -1):
    coefs_QR[i] = b[i]
    for j in range(i+1, n_features+1):
        coefs_QR[i] -= R[i, j]*coefs_QR[j]
    coefs_QR[i] /= R[i, i]

# Check if coefs obtained from normal equation and QR decomposition are same
is_same_QR = np.allclose(coefs, coefs_QR)
print(f"Are the coefs obtained from normal equation and QR decomposition same: {is_same_QR}\n")

# Save the coefs_QR to a file named coefs_QR.csv
np.savetxt('coefs_QR.csv', coefs_QR, delimiter=',')

# Find the predictions using coefs_QR
predictions_QR = np.dot(X, coefs_QR)

# Find the errors using coefs_QR
errors_QR = y-predictions_QR
rel_errors_QR = errors_QR/y
print(f"L2 norm of errors using QR decomposition: {np.linalg.norm(errors_QR)}\n")
print(f"L2 norm of relative errors using QR decomposition: {np.linalg.norm(rel_errors_QR)}\n")

# SOLVE THE NORMAL EQUATIONS USING SVD
# X = U S V^T
# We can find the inverse of X in least square sense which is known as pseudo inverse
# Normal equation: coefs = (X^T X)^{-1} X^T y = V S^+ U^T y
U, S, Vt = np.linalg.svd(X, full_matrices=False)
S_pseudo_inv = np.diag(1.0 / S) # Since S is a diagonal matrix, psuedo inverse turns out to be of this form
coefs_SVD = np.dot(np.dot(np.dot(Vt.T, S_pseudo_inv), U.T), y)

# Check if coefs obtained from normal equation and SVD are same
is_same_SVD = np.allclose(coefs, coefs_SVD)
print(f"Are the coefs obtained from normal equation and SVD same: {is_same_SVD}\n")

# Save the coefs_SVD to a file named coefs_SVD.csv
np.savetxt('coefs_SVD.csv', coefs_SVD, delimiter=',')

# Find the predictions using coefs_SVD
predictions_SVD = np.dot(X, coefs_SVD)

# Calculate the errors using coefs_SVD
errors_SVD = y-predictions_SVD
rel_errors_SVD = errors_SVD/y
print(f"L2 norm of errors using SVD: {np.linalg.norm(errors_SVD)}\n")
print(f"L2 norm of relative errors using SVD: {np.linalg.norm(rel_errors_SVD)}\n")
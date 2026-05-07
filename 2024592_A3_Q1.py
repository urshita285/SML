import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def load_mnist_npz(filepath='mnist.npz'):
    with np.load(filepath) as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test  = data['x_test']
        y_test  = data['y_test']
    return X_train, y_train, X_test, y_test

#load data
X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist_npz('mnist.npz')

#keep only classes 0,1,2
mask_train = np.isin(y_train_raw, [0,1,2])
mask_test  = np.isin(y_test_raw,  [0,1,2])
X_train_raw = X_train_raw[mask_train]
y_train_raw = y_train_raw[mask_train]
X_test_raw  = X_test_raw[mask_test]
y_test_raw  = y_test_raw[mask_test]

#flatten and normalise
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255.0
X_test  = X_test_raw.reshape(X_test_raw.shape[0], -1)   / 255.0

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Class distribution in train: {np.bincount(y_train_raw)}")
print(f"Class distribution in test: {np.bincount(y_test_raw)}")


def pca_reduce(X_train, X_test, n_components):
    mean = np.sum(X_train, axis = 0) / X_train.shape[0]
    X_centered = X_train - mean

    cov = (X_centered.T @ X_centered) / (X_train.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    U = eig_vecs[:, :n_components]
    X_train_pca = X_centered @ U
    X_test_pca = (X_test - mean) @ U
    
    return X_train_pca, X_test_pca, U, mean

p = 10
X_train_pca, X_test_pca, _, _ = pca_reduce(X_train, X_test, p)
print(f"PCA reduced to {p} components. Train shape: {X_train_pca.shape}")


#regression
def train_ridge(X, y, lam):
    n, d = X.shape
    XtX = X.T @ X
    I = np.eye(d)
    
    #regularisation
    XtX_reg = XtX + lam * I
    XtY = X.T @ y
    w = np.linalg.solve(XtX_reg, XtY)  
    return w

def predict_ridge(X, w):
    pred = X @ w
    return np.clip(pred, 0, 1)

def train_lasso(X, y, lam):
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-4)
    model.fit(X, y)
    return model.coef_

def one_vs_rest_regression(X_train, y_train, X_test, y_test, lam, method='ridge'):
    classes = [0, 1, 2]
    w_list = []
    train_preds = np.zeros((X_train.shape[0], 3))
    test_preds = np.zeros((X_test.shape[0], 3))
    
    for i, c in enumerate(classes):
        y_bin = (y_train == c).astype(float)
        
        if method == 'ridge':
            w = train_ridge(X_train, y_bin, lam)
        elif method == 'lasso':
            w = train_lasso(X_train, y_bin, lam)
        else:
            raise ValueError("method must be 'ridge' or 'lasso'")
        
        w_list.append(w)
        
        train_preds[:, i] = predict_ridge(X_train, w)
        test_preds[:, i] = predict_ridge(X_test, w)
    
    train_class = np.argmax(train_preds, axis=1)
    test_class = np.argmax(test_preds, axis=1)
    
    #compute mse
    mse_train_list = []
    mse_test_list = []
    for i, c in enumerate(classes):
        y_bin_train = (y_train == c).astype(float)
        y_bin_test = (y_test == c).astype(float)
        mse_train_list.append(np.mean((train_preds[:, i] - y_bin_train) ** 2))
        mse_test_list.append(np.mean((test_preds[:, i] - y_bin_test) ** 2))
    
    #average mse over the 3 classes
    mse_train = np.mean(mse_train_list)
    mse_test = np.mean(mse_test_list)
    
    return mse_train, mse_test, train_class, test_class, w_list


def count_nonzero_coefficients(w_list, tol=1e-6):
    nnz_per_class = [np.sum(np.abs(w) > tol) for w in w_list]
    return np.mean(nnz_per_class)


lambda_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
ridge_train_mse = []
ridge_test_mse = []
lasso_train_mse = []
lasso_test_mse = []
lasso_nnz = []
ridge_coeffs_path = []  
lasso_coeffs_path = []  

print("\n\nRunning λ sweep...\n")

for lam in lambda_vals:
    print(f"\nλ = {lam}")
    
    #ridge regression
    mse_tr, mse_te, _, _, w_list_ridge = one_vs_rest_regression(
        X_train_pca, y_train_raw, X_test_pca, y_test_raw, lam, 'ridge'
    )
    ridge_train_mse.append(mse_tr)
    ridge_test_mse.append(mse_te)
    ridge_coeffs_path.append(w_list_ridge[1])  # Store class 1 coefficients
    
    #lasso regression
    mse_tr, mse_te, _, _, w_list_lasso = one_vs_rest_regression(
        X_train_pca, y_train_raw, X_test_pca, y_test_raw, lam, 'lasso'
    )
    lasso_train_mse.append(mse_tr)
    lasso_test_mse.append(mse_te)
    lasso_coeffs_path.append(w_list_lasso[1])  # Store class 1 coefficients
    lasso_nnz.append(count_nonzero_coefficients(w_list_lasso))
    
    print(f"  Ridge - Train MSE: {mse_tr:.6f}, Test MSE: {mse_te:.6f}")
    print(f"  Lasso - Train MSE: {mse_tr:.6f}, Test MSE: {mse_te:.6f}, Non-zero coeffs: {lasso_nnz[-1]:.1f}")

#convert to arrays for plotting
ridge_coeffs_path = np.array(ridge_coeffs_path)  # shape (len(lambda), p)
lasso_coeffs_path = np.array(lasso_coeffs_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#ridge plot
axes[0].semilogx(lambda_vals, ridge_train_mse, 'b-o', label='Train MSE', linewidth=2, markersize=8)
axes[0].semilogx(lambda_vals, ridge_test_mse, 'r-o', label='Test MSE', linewidth=2, markersize=8)
axes[0].set_xlabel('λ (log scale)', fontsize=12)
axes[0].set_ylabel('Mean Squared Error', fontsize=12)
axes[0].set_title('Ridge Regression', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

#lasso plot
axes[1].semilogx(lambda_vals, lasso_train_mse, 'b-o', label='Train MSE', linewidth=2, markersize=8)
axes[1].semilogx(lambda_vals, lasso_test_mse, 'r-o', label='Test MSE', linewidth=2, markersize=8)
axes[1].set_xlabel('λ (log scale)', fontsize=12)
axes[1].set_ylabel('Mean Squared Error', fontsize=12)
axes[1].set_title('Lasso Regression', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#number of non-zero coefficients (Lasso)
plt.figure(figsize=(8, 5))
plt.semilogx(lambda_vals, lasso_nnz, 'g-o', linewidth=2, markersize=8)
plt.xlabel('λ (log scale)', fontsize=12)
plt.ylabel('Average number of non-zero coefficients', fontsize=12)
plt.title('Lasso: Feature Sparsity vs Regularization Strength', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

#regularisation paths (Class 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#ridge path
for i in range(p):
    axes[0].semilogx(lambda_vals, ridge_coeffs_path[:, i], label=f'Feat {i+1}', linewidth=1.5)
axes[0].set_xlabel('λ (log scale)', fontsize=12)
axes[0].set_ylabel('Coefficient value', fontsize=12)
axes[0].set_title('Ridge Regularization Path (Class 1)', fontsize=14)
axes[0].legend(loc='best', fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)

#lasso path
for i in range(p):
    axes[1].semilogx(lambda_vals, lasso_coeffs_path[:, i], label=f'Feat {i+1}', linewidth=1.5)
axes[1].set_xlabel('λ (log scale)', fontsize=12)
axes[1].set_ylabel('Coefficient value', fontsize=12)
axes[1].set_title('Lasso Regularization Path (Class 1)', fontsize=14)
axes[1].legend(loc='best', fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#finding best lambda
best_ridge_idx = np.argmin(ridge_test_mse)
best_lasso_idx = np.argmin(lasso_test_mse)
best_lambda_ridge = lambda_vals[best_ridge_idx]
best_lambda_lasso = lambda_vals[best_lasso_idx]

print("\n\nBest λ values:")
print(f"  Ridge: λ = {best_lambda_ridge} (Test MSE = {ridge_test_mse[best_ridge_idx]:.6f})")
print(f"  Lasso: λ = {best_lambda_lasso} (Test MSE = {lasso_test_mse[best_lasso_idx]:.6f})\n")

#vary model complexity
print("\n\nVarying PCA dimension (using best λ from above)...\n")

p_vals = [2, 5, 10, 20, 30]
ridge_train_mse_complex = []
ridge_test_mse_complex = []

for p_dim in p_vals:
    print(f"\nPCA dimension = {p_dim}")
    X_train_pca_tmp, X_test_pca_tmp, _, _ = pca_reduce(X_train, X_test, p_dim)
    mse_tr, mse_te, _, _, _ = one_vs_rest_regression(
        X_train_pca_tmp, y_train_raw, X_test_pca_tmp, y_test_raw, 
        best_lambda_ridge, 'ridge'
    )
    ridge_train_mse_complex.append(mse_tr)
    ridge_test_mse_complex.append(mse_te)
    print(f"  Train MSE: {mse_tr:.6f}, Test MSE: {mse_te:.6f}")

#plot model complexity
plt.figure(figsize=(8, 5))
plt.plot(p_vals, ridge_train_mse_complex, 'b-o', label='Train MSE', linewidth=2, markersize=8)
plt.plot(p_vals, ridge_test_mse_complex, 'r-o', label='Test MSE', linewidth=2, markersize=8)
plt.xlabel('Number of PCA components (p)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Ridge Regression: Effect of Model Complexity', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

#finding best p
best_p_idx = np.argmin(ridge_test_mse_complex)
best_p = p_vals[best_p_idx]
print(f"\nBest PCA dimension: p = {best_p} (Test MSE = {ridge_test_mse_complex[best_p_idx]:.6f})")


print("\n\nFinal Classification Accuracy\n")

#ridge with best lambda and p
X_train_pca_best, X_test_pca_best, _, _ = pca_reduce(X_train, X_test, best_p)
_, _, _, ridge_preds, _ = one_vs_rest_regression(
    X_train_pca_best, y_train_raw, X_test_pca_best, y_test_raw, 
    best_lambda_ridge, 'ridge'
)
ridge_acc = np.mean(ridge_preds == y_test_raw)

#lasso with best lambda and p
_, _, _, lasso_preds, _ = one_vs_rest_regression(
    X_train_pca_best, y_train_raw, X_test_pca_best, y_test_raw, 
    best_lambda_lasso, 'lasso'
)
lasso_acc = np.mean(lasso_preds == y_test_raw)

print(f"Ridge Regression:")
print(f"  Best λ = {best_lambda_ridge}")
print(f"  Best p = {best_p}")
print(f"  Test Classification Accuracy: {ridge_acc:.6f} ({ridge_acc*100:.2f}%)")

print(f"\nLasso Regression:")
print(f"  Best λ = {best_lambda_lasso}")
print(f"  p = {best_p}")
print(f"  Test Classification Accuracy: {lasso_acc:.6f} ({lasso_acc*100:.2f}%)")

#class-wise accuracy for the best model
print("\nClass-wise accuracy\n")
print("Ridge Regression Model")
for c in [0, 1, 2]:
    mask = (y_test_raw == c)
    class_acc = np.mean(ridge_preds[mask] == c)
    print(f"  Class {c}: {class_acc:.4f} ({class_acc*100:.2f}%)")
print("Lasso Regression Model")
for c in [0, 1, 2]:
    mask = (y_test_raw == c)
    class_acc = np.mean(lasso_preds[mask] == c)
    print(f"  Class {c}: {class_acc:.4f} ({class_acc*100:.2f}%)")
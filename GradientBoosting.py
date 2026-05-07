import numpy as np
import matplotlib.pyplot as plt

#load
def load_mnist_npz(filepath='mnist.npz'):
    with np.load(filepath) as data:
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']

X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist_npz('mnist.npz')

mask_train = (y_train_raw == 4) | (y_train_raw == 9)
mask_test  = (y_test_raw  == 4) | (y_test_raw  == 9)
X_train_raw, y_train_raw = X_train_raw[mask_train], y_train_raw[mask_train]
X_test_raw,  y_test_raw  = X_test_raw[mask_test],   y_test_raw[mask_test]
y_train_raw = np.where(y_train_raw == 4, -1, 1).astype(float)
y_test_raw  = np.where(y_test_raw  == 4, -1, 1).astype(float)

X_train = X_train_raw.reshape(len(X_train_raw), -1) / 255.0
X_test  = X_test_raw.reshape(len(X_test_raw),   -1) / 255.0

#train / val split
np.random.seed(42)
def train_val_split(X, y, n_per_class=1000):
    train_idx, val_idx = [], []
    for c in [-1, 1]:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        val_idx.extend(idx[:n_per_class])
        train_idx.extend(idx[n_per_class:])
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train_raw, 1000)

#pca
def pca_reduce(X_train, *others, n_components=5):
    mean = X_train.mean(axis=0)
    Xc   = X_train - mean
    cov  = (Xc.T @ Xc) / (len(X_train) - 1)
    vals, vecs = np.linalg.eigh(cov)
    U = vecs[:, np.argsort(vals)[::-1][:n_components]]
    return tuple((Z - mean) @ U for Z in (X_train, *others))

X_tr_pca, X_val_pca, X_test_pca = pca_reduce(X_tr, X_val, X_test, n_components=5)
print(f"Data shapes: train {X_tr_pca.shape}, val {X_val_pca.shape}, test {X_test_pca.shape}")

#thresholds
def precompute_thresholds(X, n_thresholds=1000):
    thresholds = []
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])          # sorted unique values
        mids = (vals[:-1] + vals[1:]) / 2  # true midpoints
        if len(mids) > n_thresholds:
            idx  = np.linspace(0, len(mids) - 1, n_thresholds, dtype=int)
            mids = mids[idx]
        thresholds.append(mids)
    return thresholds

THRESHOLDS = precompute_thresholds(X_tr_pca)   # compute once

#decision stump regressor
class DecisionStumpRegressor:
    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left_val  = None
        self.right_val = None

    def fit(self, X, y, thresholds):
        """Fit stump minimising SSR; thresholds is a precomputed list per feature."""
        best_ssr = np.inf
        n = len(y)

        for j, thr_list in enumerate(thresholds):
            xj = X[:, j]
            order   = np.argsort(xj)
            xj_s    = xj[order]
            y_s     = y[order]

            prefix_sum = np.cumsum(y_s)
            prefix_sq  = np.cumsum(y_s ** 2)
            total_sum  = prefix_sum[-1]
            total_sq   = prefix_sq[-1]

            split_idx = np.searchsorted(xj_s, thr_list, side='right')
            # skip degenerate splits
            valid = (split_idx > 0) & (split_idx < n)
            if not valid.any():
                continue

            si = split_idx[valid]
            l_sum = prefix_sum[si - 1]
            l_sq  = prefix_sq[si - 1]
            r_sum = total_sum - l_sum
            r_sq  = total_sq  - l_sq
            l_n   = si.astype(float)
            r_n   = (n - si).astype(float)

            ssr = (l_sq - l_sum**2 / l_n) + (r_sq - r_sum**2 / r_n)
            best_i = np.argmin(ssr)
            if ssr[best_i] < best_ssr:
                best_ssr       = ssr[best_i]
                self.feature   = j
                self.threshold = thr_list[valid][best_i]
                idx_s          = si[best_i]
                self.left_val  = l_sum[best_i] / l_n[best_i]
                self.right_val = r_sum[best_i] / r_n[best_i]

    def predict(self, X):
        left = X[:, self.feature] <= self.threshold
        out  = np.where(left, self.left_val, self.right_val)
        return out


#gradient boosting
def gradient_boosting(X_tr, y_tr, X_val, y_val, eta, T=300, thresholds=None):
    F_tr  = np.zeros(len(y_tr))
    F_val = np.zeros(len(y_val))
    mse_val   = []
    mse_train = []
    models    = []

    for t in range(T):
        residuals = y_tr - F_tr          # negative gradient of squared loss
        stump = DecisionStumpRegressor()
        stump.fit(X_tr, residuals, thresholds)
        models.append(stump)

        F_tr  += eta * stump.predict(X_tr)
        F_val += eta * stump.predict(X_val)

        mse_train.append(np.mean((y_tr  - F_tr )**2))
        mse_val.append(  np.mean((y_val - F_val)**2))

    return models, mse_train, mse_val


#run for different learnign rates
etas = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

plt.figure(figsize=(10, 6))
results = {}

for eta in etas:
    print(f"Training  eta={eta} ...", flush=True)
    models, mse_tr, mse_val = gradient_boosting(
        X_tr_pca, y_tr, X_val_pca, y_val, eta, T=300, thresholds=THRESHOLDS
    )
    best_idx = int(np.argmin(mse_val))
    print(f"  best val MSE = {mse_val[best_idx]:.6f}  at T={best_idx+1}")
    results[eta] = dict(models=models, mse_tr=mse_tr, mse_val=mse_val,
                        best_t=best_idx + 1)
    plt.plot(range(1, 301), mse_val, label=f'η={eta}')

plt.xlabel('Number of trees (T)')
plt.ylabel('Validation MSE')
plt.title('Gradient Boosting: Validation MSE vs T for different learning rates')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#best eta
best_eta = min(results, key=lambda e: results[e]['mse_val'][results[e]['best_t']-1])
best_t   = results[best_eta]['best_t']
best_models = results[best_eta]['models'][:best_t]

F_test = sum(best_eta * m.predict(X_test_pca) for m in best_models)
test_mse = np.mean((y_test_raw - F_test)**2)
print(f"\nBest η={best_eta}, T={best_t}")
print(f"Test MSE: {test_mse:.6f}")

#test results
print("\nSummary across all etas:")
print(f"{'eta':>6}  {'best_T':>6}  {'val_MSE':>10}  {'test_MSE':>10}  {'train_MSE':>10}")
for eta in etas:
    r   = results[eta]
    t   = r['best_t']
    mods = r['models'][:t]
    F   = sum(eta * m.predict(X_test_pca) for m in mods)
    tmse = np.mean((y_test_raw - F)**2)
    print(f"{eta:>6}  {t:>6}  {r['mse_val'][t-1]:>10.6f}  {tmse:>10.6f}  {r['mse_tr'][t-1]:>10.6f}")

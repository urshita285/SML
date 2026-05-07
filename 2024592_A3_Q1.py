import numpy as np
import matplotlib.pyplot as plt

#load
def load_mnist_npz(filepath='mnist.npz'):
    with np.load(filepath) as data:
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']

X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist_npz('mnist.npz')

#keep only digits 4 and 9
mask_train = (y_train_raw == 4) | (y_train_raw == 9)
mask_test  = (y_test_raw  == 4) | (y_test_raw  == 9)
X_train_raw, y_train_raw = X_train_raw[mask_train], y_train_raw[mask_train]
X_test_raw,  y_test_raw  = X_test_raw[mask_test],   y_test_raw[mask_test]

#labels, 4 to -1, 9 to +1
y_train_raw = np.where(y_train_raw == 4, -1, 1)
y_test_raw  = np.where(y_test_raw  == 4, -1, 1)

# Flatten + normalize
X_train = X_train_raw.reshape(len(X_train_raw), -1) / 255.0
X_test  = X_test_raw.reshape(len(X_test_raw),   -1) / 255.0

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

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
print(f"After split -> train {X_tr.shape}, val {X_val.shape}")

#pca
def pca_reduce(X_train, *others, n_components=5):
    mean = X_train.mean(axis=0)
    Xc   = X_train - mean
    cov  = (Xc.T @ Xc) / (len(X_train) - 1)
    vals, vecs = np.linalg.eigh(cov)
    U = vecs[:, np.argsort(vals)[::-1][:n_components]]
    return tuple((Z - mean) @ U for Z in (X_train, *others))

X_tr_pca, X_val_pca, X_test_pca = pca_reduce(X_tr, X_val, X_test, n_components=5)
print(f"PCA -> train {X_tr_pca.shape}, val {X_val_pca.shape}, test {X_test_pca.shape}")

#decision stump classifier
class DecisionStump:
    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.polarity  = 1
        self.alpha     = None

    def fit(self, X, y, w):
        n, d = X.shape
        best_err = np.inf

        for j in range(d):
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) > 1000:
                thresholds = np.linspace(vals[0], vals[-1], 1001)[:-1]
            else:
                thresholds = (vals[:-1] + vals[1:]) / 2.0

            xj = X[:, j]
            for thr in thresholds:
                # polarity = +1: predict +1 if x > thr, else -1
                pred = np.where(xj > thr, 1, -1)
                err  = w[pred != y].sum()
                # polarity = -1 is just 1 - err (flip)
                if err > 0.5:
                    err = 1.0 - err
                    pol = -1
                else:
                    pol = 1

                if err < best_err:
                    best_err       = err
                    self.feature   = j
                    self.threshold = thr
                    self.polarity  = pol

    def predict(self, X):
        xj = X[:, self.feature]
        return self.polarity * np.where(xj > self.threshold, 1, -1)


#adaboost
def adaboost(X_tr, y_tr, X_val, y_val, T=300):
    n = len(y_tr)
    w = np.full(n, 1.0 / n)

    stumps    = []
    
    val_score  = np.zeros(len(y_val))
    val_accs   = []

    for t in range(T):
        stump = DecisionStump()
        stump.fit(X_tr, y_tr, w)

        pred_tr = stump.predict(X_tr)
        err = w[pred_tr != y_tr].sum()          # already normalized (w sums to 1)
        err = np.clip(err, 1e-10, 1 - 1e-10)   # numerical safety

        alpha        = 0.5 * np.log((1 - err) / err)
        stump.alpha  = alpha
        stumps.append(stump)

        # update + renormalise weights
        w = w * np.exp(-alpha * y_tr * pred_tr)
        w /= w.sum()

        # incremental validation score update  (O(n_val) per step, not O(Tn_val))
        val_score += alpha * stump.predict(X_val)
        val_accs.append(np.mean(np.sign(val_score) == y_val))

        if (t + 1) % 50 == 0:
            print(f"  T={t+1:3d}  val_acc={val_accs[-1]:.4f}")

    return stumps, val_accs


print("\nTraining AdaBoost (T=300")
stumps, val_accs = adaboost(X_tr_pca, y_tr, X_val_pca, y_val, T=300)

#plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_accs) + 1), val_accs, linewidth=1.5)
plt.xlabel('Number of stumps (T)')
plt.ylabel('Validation accuracy')
plt.title('AdaBoost: Validation accuracy vs number of stumps')
plt.grid(True)
plt.tight_layout()
plt.show()

#best t
best_T   = int(np.argmax(val_accs)) + 1
best_val = val_accs[best_T - 1]
print(f"\nBest val accuracy: {best_val:.4f}  at T={best_T}")

#final evaluation
test_score = sum(s.alpha * s.predict(X_test_pca) for s in stumps[:best_T])
test_acc   = np.mean(np.sign(test_score) == y_test_raw)
print(f"Test accuracy with T={best_T}: {test_acc:.4f}")
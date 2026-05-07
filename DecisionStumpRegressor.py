import numpy as np
import matplotlib.pyplot as plt
import struct

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, columns = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Bad magic number in images file!")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, columns)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Bad magic number in labels file!")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load data
X_train_raw = load_images('fashion-mnist/train-images-idx3-ubyte')
y_train_raw = load_labels('fashion-mnist/train-labels-idx1-ubyte')
X_test_raw = load_images('fashion-mnist/t10k-images-idx3-ubyte')
y_test_raw = load_labels('fashion-mnist/t10k-labels-idx1-ubyte')

# Filter classes 0,1,2
mask_train = np.isin(y_train_raw, [0,1,2])
mask_test = np.isin(y_test_raw, [0,1,2])
X_train_raw = X_train_raw[mask_train]
y_train_raw = y_train_raw[mask_train]
X_test_raw = X_test_raw[mask_test]
y_test_raw = y_test_raw[mask_test]

# Flatten and normalize
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255.0
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255.0

print(f"FashionMNIST Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Class distribution: Train: {np.bincount(y_train_raw)}, Test: {np.bincount(y_test_raw)}")

# PCA reduction
def pca_reduce(X_train, X_test, n_components):
    mean = np.sum(X_train, axis=0) / X_train.shape[0]
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

# Optimized Decision Stump Regressor
class DecisionStumpRegressor:
    
    def __init__(self, n_thresholds=50):
        self.feature = None
        self.threshold = None
        self.left_pred = None
        self.right_pred = None
        self.ssr = None
        self.n_thresholds = n_thresholds  # Number of candidate thresholds to try
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_ssr = float('inf')
        best_feature = None
        best_threshold = None
        best_left_pred = None
        best_right_pred = None
        
        for feature in range(n_features):
            # Get sorted values for this feature
            values = np.sort(X[:, feature])
            
            # Use percentiles for thresholds (much faster than all unique values)
            percentiles = np.linspace(10, 90, self.n_thresholds)  # 10th to 90th percentile
            thresholds = np.percentile(values, percentiles)
            
            # Also try median and mean
            thresholds = np.unique(np.concatenate([thresholds, [np.median(values), np.mean(values)]]))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_pred = np.mean(y[left_mask])
                right_pred = np.mean(y[right_mask])
                
                left_ssr = np.sum((y[left_mask] - left_pred) ** 2)
                right_ssr = np.sum((y[right_mask] - right_pred) ** 2)
                total_ssr = left_ssr + right_ssr
                
                if total_ssr < best_ssr:
                    best_ssr = total_ssr
                    best_feature = feature
                    best_threshold = threshold
                    best_left_pred = left_pred
                    best_right_pred = right_pred
        
        self.feature = best_feature
        self.threshold = best_threshold
        self.left_pred = best_left_pred
        self.right_pred = best_right_pred
        self.ssr = best_ssr
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = X[:, self.feature] > self.threshold
        predictions[left_mask] = self.left_pred
        predictions[right_mask] = self.right_pred
        return predictions

def train_one_vs_rest_stumps(X_train, y_train, X_test, y_test):
    classes = [0, 1, 2]
    stumps = []
    train_preds = np.zeros((X_train.shape[0], 3))
    test_preds = np.zeros((X_test.shape[0], 3))
    
    for i, c in enumerate(classes):
        print(f"  Training stump for class {c}...")
        y_bin = (y_train == c).astype(float)
        
        stump = DecisionStumpRegressor(n_thresholds=30)  # Try 30 thresholds per feature
        stump.fit(X_train, y_bin)
        stumps.append(stump)
        
        train_preds[:, i] = stump.predict(X_train)
        test_preds[:, i] = stump.predict(X_test)
        
        print(f"    Class {c}: SSR = {stump.ssr:.2f}, Feature {stump.feature}, Threshold {stump.threshold:.4f}")
    
    train_class = np.argmax(train_preds, axis=1)
    test_class = np.argmax(test_preds, axis=1)
    
    mse_train_list = []
    mse_test_list = []
    for i, c in enumerate(classes):
        y_bin_train = (y_train == c).astype(float)
        y_bin_test = (y_test == c).astype(float)
        mse_train_list.append(np.mean((train_preds[:, i] - y_bin_train) ** 2))
        mse_test_list.append(np.mean((test_preds[:, i] - y_bin_test) ** 2))
    
    mse_train = np.mean(mse_train_list)
    mse_test = np.mean(mse_test_list)
    
    train_acc = np.mean(train_class == y_train)
    test_acc = np.mean(test_class == y_test)
    
    return stumps, train_preds, test_preds, mse_train, mse_test, train_acc, test_acc

# Single decision stump
print("\n\nPART 1: Single Decision Stump Regression\n")

stumps_single, train_preds, test_preds, mse_train, mse_test, train_acc, test_acc = \
    train_one_vs_rest_stumps(X_train_pca, y_train_raw, X_test_pca, y_test_raw)

print(f"\nSingle Stump Results:")
print(f"  Training MSE: {mse_train:.6f}")
print(f"  Test MSE: {mse_test:.6f}")
print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Bagging with 5 Bootstrap Samples
print("\n\nPART 2: Bagging with 5 Bootstrap Samples\n")

def bootstrap_sample(X, y, random_state=None):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    oob_indices = list(set(range(n_samples)) - set(indices))
    return X[indices], y[indices], oob_indices

n_trees = 5
bagged_stumps = []
oob_predictions = {i: [] for i in range(len(X_train_pca))}

for t in range(n_trees):
    print(f"\nTraining stump {t+1}/{n_trees}...")
    
    X_boot, y_boot, oob_idx = bootstrap_sample(X_train_pca, y_train_raw, random_state=t)
    
    boot_stumps = []
    for c in [0, 1, 2]:
        print(f"  Training class {c} stump for tree {t+1}...")
        y_bin = (y_boot == c).astype(float)
        stump = DecisionStumpRegressor(n_thresholds=30)
        stump.fit(X_boot, y_bin)
        boot_stumps.append(stump)
    bagged_stumps.append(boot_stumps)
    
    for idx in oob_idx:
        x_oob = X_train_pca[idx].reshape(1, -1)
        preds = []
        for c in [0, 1, 2]:
            preds.append(boot_stumps[c].predict(x_oob)[0])
        oob_predictions[idx].append(preds)

# OOB error
oob_true = []
oob_pred_class = []
for idx, preds_list in oob_predictions.items():
    if len(preds_list) > 0:
        avg_preds = np.mean(preds_list, axis=0)
        pred_class = np.argmax(avg_preds)
        oob_true.append(y_train_raw[idx])
        oob_pred_class.append(pred_class)

oob_error = 1 - np.mean(np.array(oob_true) == np.array(oob_pred_class))
print(f"\nAverage OOB Error: {oob_error:.4f} ({oob_error*100:.2f}%)")

# Test predictions
test_predictions = []
for x in X_test_pca:
    all_preds = []
    for boot_stumps in bagged_stumps:
        preds = []
        for c in [0, 1, 2]:
            preds.append(boot_stumps[c].predict(x.reshape(1, -1))[0])
        all_preds.append(preds)
    avg_preds = np.mean(all_preds, axis=0)
    test_predictions.append(np.argmax(avg_preds))

test_preds_array = np.array(test_predictions)
bagging_acc = np.mean(test_preds_array == y_test_raw)

# Test MSE for bagging
bagging_mse = 0
for c in [0, 1, 2]:
    y_bin_true = (y_test_raw == c).astype(float)
    y_bin_pred = np.array([np.mean([boot_stumps[c].predict(x.reshape(1, -1))[0] 
                                    for boot_stumps in bagged_stumps]) 
                          for x in X_test_pca])
    bagging_mse += np.mean((y_bin_pred - y_bin_true) ** 2)
bagging_mse /= 3

print(f"\nBagging Results:")
print(f"  Test MSE: {bagging_mse:.6f}")
print(f"  Test Accuracy: {bagging_acc:.4f} ({bagging_acc*100:.2f}%)")

# Comparison plot
print("\n\nPART 3: Comparison of Single Stump vs Bagging\n")

n_samples_to_plot = min(50, len(X_test_pca))
indices = np.random.choice(len(X_test_pca), n_samples_to_plot, replace=False)

single_preds = []
for x in X_test_pca[indices]:
    preds = [stump.predict(x.reshape(1, -1))[0] for stump in stumps_single]
    single_preds.append(np.argmax(preds))
single_preds = np.array(single_preds)

bagging_preds = []
for x in X_test_pca[indices]:
    all_preds = []
    for boot_stumps in bagged_stumps:
        preds = [boot_stumps[c].predict(x.reshape(1, -1))[0] for c in [0, 1, 2]]
        all_preds.append(preds)
    avg_preds = np.mean(all_preds, axis=0)
    bagging_preds.append(np.argmax(avg_preds))
bagging_preds = np.array(bagging_preds)

true_labels = y_test_raw[indices]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(range(n_samples_to_plot), true_labels, c='blue', marker='o', label='True Label', alpha=0.7)
axes[0].scatter(range(n_samples_to_plot), single_preds, c='red', marker='x', label='Single Stump Prediction', alpha=0.7)
axes[0].set_xlabel('Test Sample Index', fontsize=12)
axes[0].set_ylabel('Class Label (0,1,2)', fontsize=12)
axes[0].set_title('Single Decision Stump Predictions', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.5, 2.5)

axes[1].scatter(range(n_samples_to_plot), true_labels, c='blue', marker='o', label='True Label', alpha=0.7)
axes[1].scatter(range(n_samples_to_plot), bagging_preds, c='green', marker='x', label='Bagging Prediction', alpha=0.7)
axes[1].set_xlabel('Test Sample Index', fontsize=12)
axes[1].set_ylabel('Class Label (0,1,2)', fontsize=12)
axes[1].set_title('Bagging (5 Stumps) Predictions', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.show()

# Final summary
print("\n\nFINAL SUMMARY - FashionMNIST with Decision Stumps\n")
print(f"Single Decision Stump:")
print(f"Test MSE: {mse_test:.6f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\nBagging (5 stumps):")
print(f"  OOB Error: {oob_error:.4f} ({oob_error*100:.2f}%)")
print(f"  Test MSE: {bagging_mse:.6f}")
print(f"  Test Accuracy: {bagging_acc:.4f} ({bagging_acc*100:.2f}%)")

print(f"\nImprovement with Bagging:")
print(f"  MSE: {mse_test:.6f}:      {bagging_mse:.6f} ({(bagging_mse - mse_test):.6f})")
print(f"  Accuracy: {test_acc:.4f}:   {bagging_acc:.4f} ({(bagging_acc - test_acc):.4f})")

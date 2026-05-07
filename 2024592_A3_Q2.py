import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

#pca reduction
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

#decision tree
class DecisionTree:
    def __init__(self, max_depth=2, random_features=None, random_state=42):
        self.max_depth = max_depth
        self.random_features = random_features 
        self.random_state = random_state
        self.tree = None
        
    def gini(self, y):
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def weighted_gini(self, y_left, y_right):
        n_total = len(y_left) + len(y_right)
        gini_left = self.gini(y_left)
        gini_right = self.gini(y_right)
        return (len(y_left) / n_total) * gini_left + (len(y_right) / n_total) * gini_right
    
    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        
        if self.random_features is not None:
            np.random.seed(self.random_state)
            features_to_try = np.random.choice(n_features, self.random_features, replace=False)
        else:
            features_to_try = range(n_features)
        
        for feature in features_to_try:
            values = np.unique(X[:, feature])
            
            threshold = np.median(X[:, feature])
            
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            current_gini = self.weighted_gini(y[left_mask], y[right_mask])
            
            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature
                best_threshold = threshold
                best_left_idx = left_mask
                best_right_idx = right_mask
        
        return best_feature, best_threshold, best_left_idx, best_right_idx, best_gini
    
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_classes == 1:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        feature, threshold, left_idx, right_idx, gini = self.find_best_split(X, y)
        
        if feature is None:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        left_subtree = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'class': Counter(y).most_common(1)[0][0]  # Majority class at this node
        }
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, node):
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])
    
    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

#single decision tree
print("\n\nPART 1: Single Decision Tree (3 terminal nodes, height 2)\n")

#train tree with max_depth=2 (gives 3 terminal nodes)
tree = DecisionTree(max_depth=2, random_features=None)
tree.fit(X_train_pca, y_train_raw)

#predict on test set
y_pred = tree.predict(X_test_pca)

#calculate accuracies
overall_acc = np.mean(y_pred == y_test_raw)
print(f"\nOverall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

print("\nClass-wise accuracy:")
for c in [0, 1, 2]:
    mask = (y_test_raw == c)
    class_acc = np.mean(y_pred[mask] == c)
    print(f"  Class {c}: {class_acc:.4f} ({class_acc*100:.2f}%)")

#bagging
print("\n\nPART 2: Bagging with 5 Trees\n")

def bootstrap_sample(X, y, random_state=None):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    oob_indices = list(set(range(n_samples)) - set(indices))
    return X[indices], y[indices], oob_indices

n_trees = 5
trees = []
oob_predictions = {i: [] for i in range(len(X_train_pca))}  # Store OOB predictions per sample

for i in range(n_trees):
    print(f"Training tree {i+1}/{n_trees}...")
    
    X_boot, y_boot, oob_idx = bootstrap_sample(X_train_pca, y_train_raw, random_state=i)
    
    tree = DecisionTree(max_depth=2, random_features=None)
    tree.fit(X_boot, y_boot)
    trees.append(tree)
    
    for idx in oob_idx:
        pred = tree.predict_single(X_train_pca[idx], tree.tree)
        oob_predictions[idx].append(pred)

#oob error
oob_true = []
oob_pred = []
for idx, preds in oob_predictions.items():
    if len(preds) > 0:
        oob_true.append(y_train_raw[idx])
        oob_pred.append(Counter(preds).most_common(1)[0][0])

oob_error = 1 - np.mean(np.array(oob_true) == np.array(oob_pred))
print(f"\nAverage OOB Error: {oob_error:.4f} ({oob_error*100:.2f}%)")

bagging_predictions = []
for x in X_test_pca:
    votes = [tree.predict_single(x, tree.tree) for tree in trees]
    bagging_predictions.append(Counter(votes).most_common(1)[0][0])

bagging_predictions = np.array(bagging_predictions)
bagging_acc = np.mean(bagging_predictions == y_test_raw)
print(f"\nBagging Test Accuracy: {bagging_acc:.4f} ({bagging_acc*100:.2f}%)")

print("\nClass-wise accuracy (Bagging):")
for c in [0, 1, 2]:
    mask = (y_test_raw == c)
    class_acc = np.mean(bagging_predictions[mask] == c)
    print(f"  Class {c}: {class_acc:.4f} ({class_acc*100:.2f}%)")


#random forest
print("\n\nPART 3: Random Forest with Feature Subsampling\n")

k_values = [3, 4, 5, 6, 7, 8]
rf_oob_errors = []
rf_test_accuracies = []

for k in k_values:
    print(f"\nTrying k = {k} features per split...")
    
    rf_trees = []
    rf_oob_predictions = {i: [] for i in range(len(X_train_pca))}
    
    for i in range(n_trees):
        X_boot, y_boot, oob_idx = bootstrap_sample(X_train_pca, y_train_raw, random_state=i)
        
        tree = DecisionTree(max_depth=2, random_features=k, random_state=i)
        tree.fit(X_boot, y_boot)
        rf_trees.append(tree)

        for idx in oob_idx:
            pred = tree.predict_single(X_train_pca[idx], tree.tree)
            rf_oob_predictions[idx].append(pred)
    
    oob_true = []
    oob_pred = []
    for idx, preds in rf_oob_predictions.items():
        if len(preds) > 0:
            oob_true.append(y_train_raw[idx])
            oob_pred.append(Counter(preds).most_common(1)[0][0])
    
    oob_error = 1 - np.mean(np.array(oob_true) == np.array(oob_pred))
    rf_oob_errors.append(oob_error)
    
    rf_temp_predictions = []
    for x in X_test_pca:
        votes = [tree.predict_single(x, tree.tree) for tree in rf_trees]
        rf_temp_predictions.append(Counter(votes).most_common(1)[0][0])
    
    test_acc = np.mean(np.array(rf_temp_predictions) == y_test_raw)
    rf_test_accuracies.append(test_acc)
    
    print(f"  OOB Error: {oob_error:.4f}, Test Accuracy: {test_acc:.4f}")

#finding optimal k
best_k_idx = np.argmax(rf_test_accuracies)
best_k = k_values[best_k_idx]
print(f"\nOptimal k: {best_k} (Test Accuracy: {rf_test_accuracies[best_k_idx]:.4f})")

#random forest w/ optimal k
print(f"\nTraining final Random Forest with k = {best_k}...")
rf_trees_final = []
rf_oob_predictions_final = {i: [] for i in range(len(X_train_pca))}

for i in range(n_trees):
    X_boot, y_boot, oob_idx = bootstrap_sample(X_train_pca, y_train_raw, random_state=i)
    tree = DecisionTree(max_depth=2, random_features=best_k, random_state=i)
    tree.fit(X_boot, y_boot)
    rf_trees_final.append(tree)
    
    for idx in oob_idx:
        pred = tree.predict_single(X_train_pca[idx], tree.tree)
        rf_oob_predictions_final[idx].append(pred)

#final oob error
oob_true = []
oob_pred = []
for idx, preds in rf_oob_predictions_final.items():
    if len(preds) > 0:
        oob_true.append(y_train_raw[idx])
        oob_pred.append(Counter(preds).most_common(1)[0][0])

final_oob_error = 1 - np.mean(np.array(oob_true) == np.array(oob_pred))
print(f"Final OOB Error: {final_oob_error:.4f} ({final_oob_error*100:.2f}%)")

#final test accuracy
rf_final_predictions = []
for x in X_test_pca:
    votes = [tree.predict_single(x, tree.tree) for tree in rf_trees_final]
    rf_final_predictions.append(Counter(votes).most_common(1)[0][0])

rf_final_predictions = np.array(rf_final_predictions)
rf_final_acc = np.mean(rf_final_predictions == y_test_raw)
print(f"Random Forest Test Accuracy: {rf_final_acc:.4f} ({rf_final_acc*100:.2f}%)")

print("\nClass-wise accuracy (Random Forest):")
for c in [0, 1, 2]:
    mask = (y_test_raw == c)
    class_acc = np.mean(rf_final_predictions[mask] == c)
    print(f"  Class {c}: {class_acc:.4f} ({class_acc*100:.2f}%)")

print("\n\nCOMPARISON SUMMARY")

# Store single tree accuracy
single_tree_acc = np.mean(y_pred == y_test_raw)

print(f"Single Decision Tree (height 2):    Accuracy = {single_tree_acc:.4f} ({single_tree_acc*100:.2f}%)")
print(f"Bagging (5 trees):                  Accuracy = {bagging_acc:.4f} ({bagging_acc*100:.2f}%)")
print(f"Random Forest (k={best_k}):                Accuracy = {rf_final_acc:.4f} ({rf_final_acc*100:.2f}%)")

print("\nRandom Forest vs Bagging:")

if rf_final_acc > bagging_acc:
    improvement = rf_final_acc - bagging_acc
    print(f"Random Forest performs BETTER than Bagging")
    print(f"  Improvement: +{improvement:.4f} (+{improvement*100:.2f}%)")
    print("  Reason: Feature subsampling reduces correlation between trees,")
    print("  leading to lower variance and better generalization.")
elif rf_final_acc < bagging_acc:
    improvement = bagging_acc - rf_final_acc
    print(f"Random Forest performs WORSE than Bagging")
    print(f"  Difference: -{improvement:.4f} (-{improvement*100:.2f}%)")
    print("  Possible reason: p=10 is small, so subsampling may lose important features.")
else:
    print("Random Forest and Bagging have EQUAL accuracy")
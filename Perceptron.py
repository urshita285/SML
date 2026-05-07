import numpy as np
import matplotlib.pyplot as plt

#generate dataset
np.random.seed(42)
def generate_dataset(mean1, mean2, cov, n_samples=200):
    X1 = np.random.multivariate_normal(mean1, cov, n_samples)
    X2 = np.random.multivariate_normal(mean2, cov, n_samples)
    X = np.vstack([X1, X2])
    y = np.hstack([-np.ones(n_samples), np.ones(n_samples)])
    return X, y

#dataset a
covA = np.eye(2)
X_A, y_A = generate_dataset([-3,-3], [3,3], covA, 200)

#dataset b
covB = 3 * np.eye(2)
X_B, y_B = generate_dataset([-3,-3], [3,3], covB, 200)

#validation split
def train_val_split(X, y, val_ratio=0.3):
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_train_A, X_val_A, y_train_A, y_val_A = train_val_split(X_A, y_A, 0.3)
X_train_B, X_val_B, y_train_B, y_val_B = train_val_split(X_B, y_B, 0.3)

#perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=300):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.misclassified_hist = []

    def fit(self, X, y, X_val, y_val):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        converged_epoch = -1
        for epoch in range(self.max_epochs):
            misclass = 0
            for i in range(n_samples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    misclass += 1
            self.misclassified_hist.append(misclass)
            if misclass == 0:
                converged_epoch = epoch + 1
                break
        return converged_epoch

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

#run for both datasets
def run_perceptron(X_train, y_train, X_val, y_val, title):
    perc = Perceptron(learning_rate=0.01, max_epochs=300)
    conv_epoch = perc.fit(X_train, y_train, X_val, y_val)
    # Validation accuracy
    y_pred_val = perc.predict(X_val)
    val_acc = np.mean(y_pred_val == y_val)
    # Plot misclassified per epoch
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(perc.misclassified_hist)+1), perc.misclassified_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Number of misclassifications')
    plt.title(f'{title} - Misclassifications per epoch')
    plt.grid(True)
    # Plot decision boundary
    plt.subplot(1,2,2)
    # Plot training and validation data
    plt.scatter(X_train[y_train==-1,0], X_train[y_train==-1,1], c='blue', marker='o', label='train -1', alpha=0.5)
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='red', marker='o', label='train +1', alpha=0.5)
    plt.scatter(X_val[y_val==-1,0], X_val[y_val==-1,1], c='cyan', marker='x', label='val -1', alpha=0.7)
    plt.scatter(X_val[y_val==1,0], X_val[y_val==1,1], c='orange', marker='x', label='val +1', alpha=0.7)
    # Plot decision line
    x_vals = np.linspace(-6, 6, 100)
    if perc.w[1] != 0:
        y_vals = -(perc.w[0]*x_vals + perc.b) / perc.w[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    else:
        # vertical line
        x_line = -perc.b / perc.w[0]
        plt.axvline(x=x_line, color='k', linestyle='--', label='Decision boundary')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'{title} - Decision boundary')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"{title}: Converged at epoch {conv_epoch if conv_epoch>0 else 'not converged'}")
    print(f"Validation accuracy: {val_acc:.4f}")
    return conv_epoch, val_acc

print("Dataset A (covariance I):")
conv_A, acc_A = run_perceptron(X_train_A, y_train_A, X_val_A, y_val_A, "Dataset A")
print("\nDataset B (covariance 3I):")
conv_B, acc_B = run_perceptron(X_train_B, y_train_B, X_val_B, y_val_B, "Dataset B")

#final results
print("\nObservation: Dataset A (small variance) is linearly separable with perceptron; converges quickly.")
print("Dataset B (larger variance) has more overlap, may not converge or requires more epochs.")

import numpy as np
import matplotlib.pyplot as plt
import struct
from sklearn.manifold import TSNE

def load_mnist_npz(filepath='mnist.npz'):
    with np.load(filepath) as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test  = data['x_test']
        y_test  = data['y_test']
    return X_train, y_train, X_test, y_test

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, columns = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Bad magic number in labels file!")
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

def sample_per_class(X, y, n = 100):
    np.random.seed(1)
    Xs, ys = [], []

    for c in [0, 1, 2]:
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, n, replace=False)
        Xs.append(X[chosen])
        ys.append(y[chosen])

    return np.vstack(Xs), np.hstack(ys)

def compute_mle(X, y):
    means = {}
    covs = {}

    for c in [0, 1, 2]:
        Xc = X[y == c]
        mu = np.mean(Xc, axis=0)

        centered = Xc - mu
        cov = (centered.T @ centered) / Xc.shape[0]
        cov += 1e-6 * np.eye(cov.shape[0])

        means[c] = mu
        covs[c] = cov

    return means, covs

def lda_predict(X, means, covs):
    combined_cov = (covs[0] + covs[1] + covs[2]) / 3
    inv_cov = np.linalg.pinv(combined_cov)

    prior = 1/3
    log_prior = np.log(prior)

    class_terms = {}
    for c in [0, 1, 2]:
        mu = means[c]

        const_term = -0.5 * mu @ inv_cov @ mu + log_prior
        linear_coef = inv_cov @ mu 
        class_terms[c] = (linear_coef, const_term)

    preds = []
    for x in X:
        scores = []
        for c in [0, 1, 2]:
            linear_coef, const_term = class_terms[c]
            g = x @ linear_coef + const_term
            scores.append(g)
        preds.append(np.argmax(scores))

    return np.array(preds)

def qda_predict(X, means, covs):
    prior = 1/3
    log_prior = np.log(prior)

    inv_covs = {}
    log_dets = {}
    const_terms = {}
    
    for c in [0, 1, 2]:
        sigma = covs[c] + np.eye(covs[c].shape[0]) * 1e-8  # Regularized
        inv_cov = np.linalg.pinv(sigma)
        sign, logdet = np.linalg.slogdet(sigma)
        
        inv_covs[c] = inv_cov
        log_dets[c] = logdet
        const_terms[c] = -0.5 * logdet + log_prior

    preds = []
    for x in X:
        scores = []
        for c in [0, 1, 2]:
            diff = x - means[c]
            inv_cov = inv_covs[c]
            
            mahalanobis = diff @ inv_cov @ diff

            g = -0.5 * mahalanobis + const_terms[c]
            scores.append(g)
        preds.append(np.argmax(scores))

    return np.array(preds)

def plot_tsne(X_train, y_train, X_test, y_test):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    X_combined = np.vstack([X_train, X_test])
    X_tsne = tsne.fit_transform(X_combined)
    
    n_train = len(X_train)
    X_train_tsne = X_tsne[:n_train]
    X_test_tsne = X_tsne[n_train:]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for c in [0, 1, 2]:
        mask = (y_train == c)
        axes[0].scatter(
            X_train_tsne[mask, 0],
            X_train_tsne[mask, 1],
            label=f'Digit {c}',
            alpha=0.7,
            s=30
        )
    axes[0].set_title('Training Set (300 samples: 100 per digit)')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for c in [0, 1, 2]:
        mask = (y_test == c)
        axes[1].scatter(
            X_test_tsne[mask, 0],
            X_test_tsne[mask, 1],
            label=f'Digit {c}',
            alpha=0.7,
            s=30
        )
    axes[1].set_title('Test Set (300 samples: 100 per digit)')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    X_train_img, y_train_all, X_test_img, y_test_all  = load_mnist_npz('mnist.npz')

    train_mask = (y_train_all <= 2)
    test_mask  = (y_test_all  <= 2)

    X_train_img = X_train_img[train_mask]
    y_train_all = y_train_all[train_mask]

    X_test_img  = X_test_img[test_mask]
    y_test_all  = y_test_all[test_mask]

    X_train_all = X_train_img.reshape(X_train_img.shape[0], -1) / 255.0
    X_test_all  = X_test_img.reshape(X_test_img.shape[0], -1) / 255.0

    X_train, y_train = sample_per_class(X_train_all, y_train_all, 100)
    X_test,  y_test  = sample_per_class(X_test_all,  y_test_all,  100)

    means, covs = compute_mle(X_train, y_train)

    lda_preds = lda_predict(X_test, means, covs)
    qda_preds = qda_predict(X_test, means, covs)

    print("LDA Accuracy:", accuracy(y_test, lda_preds))
    print("QDA Accuracy:", accuracy(y_test, qda_preds))

    plot_tsne(X_train, y_train, X_test, y_test)


    sample_idx = 0
    x_sample = X_test[sample_idx]
    true_label = y_test[sample_idx]
    
    print("\n\nDiscriminant values for test sample #", sample_idx, " (True label: ", true_label, "):")
    
    prior = 1/3
    log_prior = np.log(prior)
    
    print("\nQDA Discriminant Values (higher is better):")
    for c in [0, 1, 2]:
        diff = x_sample - means[c]
        sigma = covs[c] + np.eye(covs[c].shape[0]) * 1e-8
        inv_cov = np.linalg.pinv(sigma)
        mahalanobis = diff @ inv_cov @ diff
        sign, logdet = np.linalg.slogdet(sigma)
        g = -0.5 * mahalanobis - 0.5 * logdet + log_prior
        print(f"Class {c}: {g:.4f}")
    
    print("\nLDA Discriminant Values (higher is better):")
    combined_cov = (covs[0] + covs[1] + covs[2]) / 3
    inv_combined = np.linalg.pinv(combined_cov)
    for c in [0, 1, 2]:
        g = x_sample @ inv_combined @ means[c] - 0.5 * means[c] @ inv_combined @ means[c] + log_prior
        print(f"Class {c}: {g:.4f}")
    
    print(f"\nLDA predicted: {lda_preds[sample_idx]}")
    print(f"QDA predicted: {qda_preds[sample_idx]}")

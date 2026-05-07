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
        mu = np.sum(Xc, axis=0) / Xc.shape[0]

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

def pca_reduce(X_train, X_test, variance_retain = 0.75):
    n, d = X_train.shape

    mean = np.sum(X_train, axis = 0) / X_train.shape[0]
    X_centered = X_train - mean

    cov = (X_centered.T @ X_centered) / (n - 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    total_var = np.sum(eig_vals)

    cum_var = np.cumsum(eig_vals) / total_var
    if variance_retain >= 1.0:
        k = d
    else:
        k = np.searchsorted(cum_var, variance_retain) + 1 

    k = min(k,d)

    U_p = eig_vecs[:, :k] 

    X_train_pca = X_centered @ U_p

    X_test_centered = X_test - mean
    X_test_pca = X_test_centered @ U_p
    exp_var = cum_var[k-1]

    print(f"Retained {variance_retain*100:.0f}% variance with {k} components (actual: {exp_var*100:.2f}%)")

    return X_train_pca, X_test_pca, U_p, mean, exp_var

def fda_reduce(X_train, y_train, X_test, n_components=2):
    classes = np.unique(y_train)

    d = X_train.shape[1]
    overall_mean = np.mean(X_train, axis=0)
    S_W = np.zeros((d, d))
    S_B = np.zeros((d, d))

    for c in classes:
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]
        mu_c = np.mean(X_c, axis=0)
        
        centered = X_c - mu_c
        S_W += centered.T @ centered
        
        diff = (mu_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (diff @ diff.T)
    
    S_W += 1e-6 * np.eye(d)

    S_W_inv = np.linalg.pinv(S_W)
    S = S_W_inv @ S_B
    
    eig_vals, eig_vecs = np.linalg.eigh(S) 
    
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    W = eig_vecs[:, :n_components]
    
    X_train_fda = X_train @ W
    X_test_fda = X_test @ W

    return X_train_fda, X_test_fda, W

def plot_2d_projection(X_train, y_train, X_test, y_test, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for c in [0, 1, 2]:
        mask_train = (y_train == c)
        axes[0].scatter(X_train[mask_train, 0], X_train[mask_train, 1],
                        label=f'Digit {c}', alpha=0.7, s=30)
    axes[0].set_title(f'{title} - Training Set')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for c in [0, 1, 2]:
        mask_test = (y_test == c)
        axes[1].scatter(X_test[mask_test, 0], X_test[mask_test, 1],
                        label=f'Digit {c}', alpha=0.7, s=30)
    axes[1].set_title(f'{title} - Test Set')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def show_reconstruction(X_test, X_test_pca, U_p, mean, y_test, n_samples=5):
    X_test_recon = mean + (X_test_pca @ U_p.T)
    
    print(f"\nReconstruction MSE for first {n_samples} test samples:")
    for i in range(min(n_samples, len(X_test))):
        mse = np.mean((X_test[i] - X_test_recon[i])**2)
        print(f"  Sample {i} (true label: {y_test[i]}): MSE = {mse:.6f}")
    
    if n_samples > 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(X_test[0].reshape(28, 28), cmap='gray')
        axes[0].set_title(f'Original (Digit {y_test[0]})')
        axes[1].imshow(X_test_recon[0].reshape(28, 28), cmap='gray')
        axes[1].set_title('Reconstructed')
        plt.tight_layout()
        plt.show()


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

    print("\nOriginal features (784 dimensions)")

    means, covs = compute_mle(X_train, y_train)

    lda_preds = lda_predict(X_test, means, covs)
    qda_preds = qda_predict(X_test, means, covs)

    print("LDA Accuracy:", accuracy(y_test, lda_preds))
    print("QDA Accuracy:", accuracy(y_test, qda_preds))



    print("\n\nPCA Dimensionality Reduction")
    variances = [0.75, 0.9, '2comp']

    for var in variances:
        if var == '2comp':
            print("\nPCA with 2 components")

            X_train_pca, X_test_pca, U_p, mean, exp_var = pca_reduce(X_train, X_test, variance_retain = 1)
            X_train_pca = X_train_pca[:, :2]
            X_test_pca = X_test_pca[:, :2]
            print(f"Using first 2 principal components")
        else:
            print(f"\nPCA with {var*100:.0f}% variance retained")
            X_train_pca, X_test_pca, U_p, mean, exp_var = pca_reduce(X_train, X_test, variance_retain=var)
        
        means_pca, covs_pca = compute_mle(X_train_pca, y_train)
        
        lda_preds_train = lda_predict(X_train_pca, means_pca, covs_pca)
        qda_preds_train = qda_predict(X_train_pca, means_pca, covs_pca)
        lda_preds_test = lda_predict(X_test_pca, means_pca, covs_pca)
        qda_preds_test = qda_predict(X_test_pca, means_pca, covs_pca)
        
        print(f"LDA Train Accuracy: {accuracy(y_train, lda_preds_train):.4f}")
        print(f"LDA Test Accuracy : {accuracy(y_test, lda_preds_test):.4f}")
        print(f"QDA Train Accuracy: {accuracy(y_train, qda_preds_train):.4f}")
        print(f"QDA Test Accuracy : {accuracy(y_test, qda_preds_test):.4f}")

        if var == 0.75:
            show_reconstruction(X_test, X_test_pca, U_p, mean, y_test, n_samples=5)

        if X_train_pca.shape[1] == 2:
            plot_2d_projection(X_train_pca, y_train, X_test_pca, y_test, f"PCA ({var if var!='2comp' else 2} components)")

    print("\n\nFisher Discriminant Analysis\n")
    print("FDA with 2 components (max for 3 classes)")

    X_train_fda, X_test_fda, W = fda_reduce(X_train, y_train, X_test, n_components=2)
    
    means_fda, covs_fda = compute_mle(X_train_fda, y_train)
    
    lda_preds_train = lda_predict(X_train_fda, means_fda, covs_fda)
    qda_preds_train = qda_predict(X_train_fda, means_fda, covs_fda)
    lda_preds_test = lda_predict(X_test_fda, means_fda, covs_fda)
    qda_preds_test = qda_predict(X_test_fda, means_fda, covs_fda)
    
    print(f"LDA Train Accuracy: {accuracy(y_train, lda_preds_train):.4f}")
    print(f"LDA Test Accuracy : {accuracy(y_test, lda_preds_test):.4f}")
    print(f"QDA Train Accuracy: {accuracy(y_train, qda_preds_train):.4f}")
    print(f"QDA Test Accuracy : {accuracy(y_test, qda_preds_test):.4f}")
    
    plot_2d_projection(X_train_fda, y_train, X_test_fda, y_test, "FDA (2 components)")

    print("\n\nt-SNE Visualisation\n")
    plot_tsne(X_train, y_train, X_test, y_test)

    print("Discriminant Values for a Sample Test Point")
    
    sample_idx = 0
    x_sample = X_test[sample_idx]
    true_label = y_test[sample_idx]
    
    print("Discriminant values for test sample #", sample_idx, " (True label: ", true_label, "):")
    
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
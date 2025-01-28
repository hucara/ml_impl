import numpy as np


def compute_pca(A_centered: np.array):
    # Compute SVD
    ATA = A_centered.T @ A_centered

    eigenvals, V = np.linalg.eigh(ATA)

    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    V = V[:, idx]

    S = np.sqrt(np.abs(eigenvals))
    Sigma = np.diag(S)

    U = A_centered @ V @ np.linalg.inv(Sigma)

    # Get principal components
    principal_components = A_centered @ V

    # Calculate explained variance
    total_variance = np.sum(eigenvals)
    explained_variance_ratio = eigenvals / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    return {'principal_components': principal_components, 'total_variance': total_variance,
            'explained_variance_ratio': explained_variance_ratio, 
            'cumulative_variance_ratio': cumulative_variance_ratio}
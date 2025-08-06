# File: rd_openmax.py
import numpy as np
from scipy.stats import weibull_min
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinvh  # Pseudo-inverse for stability
import heapq
import multiprocessing as mp
import warnings

def revt_scaling(t, a=9.68426338, b=1.35437019, c=2.19824087):
    return a * np.power(t, -b) + c

def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

def fit_weibull_mahalanobis(class_features, class_means, class_inv_covs, tailsize=45, use_revt=True):
    """
    Fit Weibull models using Mahalanobis distance instead of Euclidean.
    Args:
        class_features (dict): {class_idx: [feature1, feature2, ...]}
        class_means (dict): {class_idx: mean_vector}
        class_inv_covs (dict): {class_idx: inverse_covariance_matrix}
    Returns:
        dict: {class_idx: (shape, loc, scale)}
    """
    weibull_models = {}

    for cls, features in class_features.items():
        inv_cov = class_inv_covs[cls]
        mu = class_means[cls]
        distances = [mahalanobis(f, mu, inv_cov) for f in features]
        tail = sorted(distances)[-tailsize:]

        if np.std(tail) < 1e-6:
            print(f"[WARNING] Class {cls} tail too flat → fallback")
            shape, loc, scale = 1.0, 0.0, 1.0
        else:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
            if use_revt:
                c = revt_scaling(tailsize)
                shape /= c

        weibull_models[cls] = (shape, loc, scale)
    return weibull_models


def openmax_predict_mahalanobis(z, class_means, class_inv_covs, weibull_models):
    modified_scores = []
    unknown_score = 0.0

    for cls, mu in class_means.items():
        inv_cov = class_inv_covs[cls]
        dist = mahalanobis(z, mu, inv_cov)

        shape, loc, scale = weibull_models[cls]
        try:
            wscore = weibull_min.cdf(dist, shape, loc, scale)
        except:
            wscore = 1.0

        modified_scores.append(1 - wscore)
        unknown_score += wscore

    total = sum(modified_scores) + unknown_score
    if total == 0 or np.isnan(total):
        probs = [1.0 / len(modified_scores)] * len(modified_scores)
        unk_prob = 0.0
    else:
        probs = [s / total for s in modified_scores]
        unk_prob = unknown_score / total

    unk_prob = np.clip(unk_prob, 0.0, 1.0)
    return np.argmax(probs), unk_prob


def extract_class_stats_mahalanobis(model, classifier, loader, device):
    model.eval()
    if classifier is not None:
        classifier.eval()

    class_features = {}

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            labels = labels.numpy()

            if classifier is not None:
                feats = model(images)
                feats = F.normalize(feats, dim=1)
            else:
                feats = model(images)

            feats = feats.cpu().numpy()
            for f, y in zip(feats, labels):
                class_features.setdefault(y, []).append(f)

    class_means = {}
    class_inv_covs = {}
    for cls, feats in class_features.items():
        feats = np.array(feats)
        mu = feats.mean(axis=0)
        cov = np.cov(feats.T) + 1e-6 * np.eye(feats.shape[1])  # Regularization
        inv_cov = pinvh(cov)

        class_means[cls] = mu
        class_inv_covs[cls] = inv_cov

    return class_features, class_means, class_inv_covs


def extract_encoder_prototypes(model, loader, device):
    """
    Tính prototype từ output của encoder (không qua projection head).
    """
    model.eval()
    class_features = {}

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            labels = labels.numpy()

            feats = model.encoder(images)  # output 8256-dim
            feats = F.normalize(feats, dim=1)  # optional nếu dùng cosine
            feats = feats.cpu().numpy()

            for f, y in zip(feats, labels):
                class_features.setdefault(y, []).append(f)

    class_means = {
        cls: np.mean(np.stack(feats), axis=0)
        for cls, feats in class_features.items()
    }

    return class_means  # {class_id: mean_vector}


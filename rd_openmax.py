# File: rd_openmax.py
import numpy as np
from scipy.stats import weibull_min
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinvh  # Pseudo-inverse for stability

def fit_weibull_rd(class_features, class_means, tailsize=20, use_revt=True):
    """
    Fit a Weibull distribution to the tail distances between each class's features and its mean.
    Args:
        class_features (dict): {class_idx: [feature1, feature2, ...]}
        class_means (dict): {class_idx: mean_vector}
        tailsize (int): Number of largest distances to use for fitting
    Returns:
        dict: {class_idx: (shape, loc, scale)}
    """
    weibull_models = {}
    def revt_scaling(t, a=9.68426338, b=1.35437019, c=2.19824087):
        return a * np.power(t, -b) + c

    for cls, features in class_features.items():
        distances = [np.linalg.norm(f - class_means[cls]) for f in features]
        tail = sorted(distances)[-tailsize:]

        shape, loc, scale = weibull_min.fit(tail, floc=0)

        if use_revt:
            c = revt_scaling(tailsize)
            shape /= c  # Apply REVT correction

        weibull_models[cls] = (shape, loc, scale)
    return weibull_models

def openmax_predict_rd(z, class_means, weibull_models, return_unk_prob=False, threshold=0.5):
    """
    Apply RD-OpenMax scoring to classify known/unknown.
    Args:
        z (np.ndarray): feature vector from encoder
        class_means (dict): {class_idx: mean_vector}
        weibull_models (dict): {class_idx: (shape, loc, scale)}
    Returns:
        int: predicted class or -1 for unknown
    """
    modified_scores = []
    unknown_score = 0.0

    for cls, mean in class_means.items():
        dist = np.linalg.norm(z - mean)
        shape, loc, scale = weibull_models[cls]
        wscore = weibull_min.cdf(dist, shape, loc, scale)
        modified_scores.append(1 - wscore)
        unknown_score += wscore

    total = sum(modified_scores) + unknown_score
    probs = [s / total for s in modified_scores]
    unk_prob = unknown_score / total

    return np.argmax(probs), unk_prob

def extract_class_stats(model, classifier, loader, device):
    """
    Trích xuất đặc trưng lớp và mean vector cho RD-OpenMax.
    Nếu có classifier, dùng output của encoder + classifier.
    Nếu không có, chỉ dùng output từ model (giả sử đã có logits).
    """
    model.eval()
    if classifier is not None:
        classifier.eval()

    class_features = {}
    class_means = {}

    with torch.no_grad():
        for images, labels in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            labels = labels.numpy()

            if classifier is not None:
                feats = model.encoder(images)          # [B, 1280]
                feats = F.normalize(feats, dim=1)
            else:
                feats = model(images)                # [B, feat_dim or logits]

            feats = feats.cpu().numpy()

            for f, y in zip(feats, labels):
                class_features.setdefault(y, []).append(f)

    for cls, feats in class_features.items():
        class_means[cls] = np.mean(feats, axis=0)

    return class_features, class_means


## For 2 phases training

def fit_weibull_mahalanobis(class_features, tailsize=20, use_revt=True):
    """
    Fit Weibull models per class using Mahalanobis distance.
    Returns:
        - weibull_models: dict of {class: {mean, inv_cov, shape, loc, scale}}
    """
    weibull_models = {}

    def revt_scaling(t, a=9.68426338, b=1.35437019, c=2.19824087):
        return a * np.power(t, -b) + c

    for cls, features in class_features.items():
        features = np.array(features)
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-6
        inv_cov = pinvh(cov)

        distances = [mahalanobis(f, mean, inv_cov) for f in features]
        tail = sorted(distances)[-tailsize:]

        shape, loc, scale = weibull_min.fit(tail, floc=0)
        if use_revt:
            c = revt_scaling(tailsize)
            shape /= c

        weibull_models[cls] = {
            'mean': mean,
            'inv_cov': inv_cov,
            'shape': shape,
            'loc': loc,
            'scale': scale
        }

    return weibull_models

# === 2. Update `openmax_predict_rd` to Mahalanobis-based ===
def openmax_predict_mahalanobis(z, weibull_models):
    min_dist = float('inf')
    best_class = -1
    unk_score = 0.0
    modified_scores = []

    for cls, model in weibull_models.items():
        mean = model['mean']
        inv_cov = model['inv_cov']
        shape, loc, scale = model['shape'], model['loc'], model['scale']

        dist = mahalanobis(z, mean, inv_cov)
        prob = weibull_min.cdf(dist, shape, loc=loc, scale=scale)

        modified_scores.append(1 - prob)
        unk_score += prob

        if dist < min_dist:
            min_dist = dist
            best_class = cls

    total = sum(modified_scores) + unk_score
    probs = [s / total for s in modified_scores]
    unk_prob = unk_score / total

    return np.argmax(probs), unk_prob


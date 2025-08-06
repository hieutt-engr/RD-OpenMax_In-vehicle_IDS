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

def fit_weibull_rd(class_features, class_means, tailsize=45, use_revt=True):
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

        # Kiểm tra tail validity
        if np.std(tail) < 1e-6:
            print(f"[WARNING] Class {cls} tail is too flat; skipping Weibull fitting")
            shape, loc, scale = 1.0, 0.0, 1.0  # fallback
        else:
            shape, loc, scale = weibull_min.fit(tail, floc=0)

            if use_revt:
                c = revt_scaling(tailsize)
                shape /= c

        weibull_models[cls] = (shape, loc, scale)
    return weibull_models


def openmax_predict_rd(z, class_means, weibull_models):
    modified_scores = []
    unknown_score = 0.0

    for cls, mean in class_means.items():
        dist = np.linalg.norm(z - mean)
        shape, loc, scale = weibull_models[cls]

        try:
            wscore = weibull_min.cdf(dist, shape, loc, scale)
        except Exception as e:
            print(f"[WARNING] Weibull CDF failed for class {cls}, dist={dist:.4f}")
            wscore = 1.0  # fallback

        modified_scores.append(1 - wscore)
        unknown_score += wscore

    total = sum(modified_scores) + unknown_score

    if total == 0 or np.isnan(total):
        probs = [1.0 / len(modified_scores)] * len(modified_scores)  # uniform fallback
        unk_prob = 0.0
    else:
        probs = [s / total for s in modified_scores]
        unk_prob = unknown_score / total

    # Clamp final output
    unk_prob = np.clip(unk_prob, 0.0, 1.0)
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


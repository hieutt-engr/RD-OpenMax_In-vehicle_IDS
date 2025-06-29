import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EfficientNetV2_Embedding(nn.Module):
    def __init__(self, model_name='efficientnetv2_m', embedding_dim=1280, pretrained=True):
        super(EfficientNetV2_Embedding, self).__init__()

        # Load EfficientNetV2 from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Compute the final output of the backbone
        self.out_channels = self.backbone.feature_info[-1]['num_chs']  # Usually 1280 for efficientnetv2_s

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection layer (embedding)
        self.fc = nn.Linear(self.out_channels, embedding_dim)

    def forward(self, x):
        # Pass through the backbone
        features = self.backbone(x)[-1]  # Get the final output [B, C, H, W]

        # Global pooling
        features = self.pool(features)           # [B, C, 1, 1]
        features = features.view(features.size(0), -1)  # [B, C]

        # Projection
        embedding = self.fc(features)  # [B, embedding_dim]
        return embedding

class ConEfficientNetV2(nn.Module):
    def __init__(self, model_name='efficientnetv2_s', embedding_dim=1280, feat_dim=128, head='mlp', pretrained=True):
        super(ConEfficientNetV2, self).__init__()
        
        self.encoder = EfficientNetV2_Embedding(model_name=model_name, embedding_dim=embedding_dim, pretrained=pretrained)

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, feat_dim)
            )
        elif head == 'linear':
            self.head = nn.Linear(embedding_dim, feat_dim)
        else:
            raise NotImplementedError(f"Projection head '{head}' not supported.")

    def forward(self, x):
        embedding = self.encoder(x)  # [B, embedding_dim]
        feat = F.normalize(self.head(embedding), dim=1)  # [B, feat_dim]
        return feat

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=5):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features, return_embeddings=False):
        if return_embeddings:
            return features
        return self.fc(features)
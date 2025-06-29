import torch
import torch.nn as nn
import warnings as warn
from networks.resnet import *
from networks.mpncovresnet import *
from networks.efficient_net_custom import *

def get_basemodel(modeltype, input_size, attention, pretrained=False):
    if input_size < 128:
        modeltype = modeltype + 'tiny'
    model = globals()[modeltype]
    if pretrained == False:
        warn.warn('You will use model that randomly initialized!')

    return model(pretrained=pretrained, input_size=input_size, attention=attention)


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three parts:
       1) feature extractor
       2) global image representation
       3) classifier
    """

    def __init__(self, modeltype, input_size, attention, pretrained=False):
        super(Basemodel, self).__init__()
        self.input_size = input_size
        self.modeltype = modeltype
        self.pretrained = pretrained

        basemodel = get_basemodel(modeltype, input_size, attention, pretrained)

        if modeltype.startswith('resnet'):
            basemodel = self._reconstruct_resnet(basemodel)
        elif modeltype.startswith('mpncovresnet'):
            basemodel = self._reconstruct_mpncovresnet(basemodel)

        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim
        
    def _reconstruct_resnet(self, basemodel):
        model = nn.Module()
        model.features = nn.Sequential(*list(basemodel.children())[:-2])
        model.representation = basemodel.avgpool
        model.classifier = basemodel.fc
        model.representation_dim = basemodel.fc.weight.size(1)
        return model

    def _reconstruct_mpncovresnet(self, basemodel):
        model = nn.Module()
        if self.pretrained:
            model.features = nn.Sequential(*list(basemodel.children())[:-1])
            model.representation_dim = basemodel.layer_reduce.weight.size(0)
        else:
            model.features = nn.Sequential(*list(basemodel.children())[:-4])
            model.representation_dim = basemodel.layer_reduce.weight.size(1)
        model.representation = None
        model.classifier = basemodel.fc
        return model

    def forward(self, x):

        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

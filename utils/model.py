import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_class, model_name, model_weights):
        super(Classifier, self).__init__()
        if model_name=='alexnet':
            net = models.alexnet(pretrained=True)
            self.features = net.features
            self.fc = nn.Sequential(nn.Linear(9216, num_class))
        elif model_name=='efficientnet':
            if model_weights is None:
                net = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            else:
                net = models.efficientnet_v2_s()
                net.load_state_dict(torch.load(model_weights))
        
            self.features = net.features
            self.fc = nn.Sequential(nn.Linear(62720, num_class))
        else:    
            self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1),
                                          nn.Conv2d(8, 16, 3, stride=2, padding=1),
                                          nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                          nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                         nn.Conv2d(64, 128, 3, stride=2, padding=1))
            self.fc = nn.Sequential(nn.Linear(6272, num_class))
            
        for param in self.features.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        output = self.features(x)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output
        

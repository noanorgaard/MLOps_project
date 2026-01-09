import torch
import torch.nn as nn
from torchvision import models

class AIOrNotClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        """
        Initializes the ResNet18 model for binary classification.
        """
        super(AIOrNotClassifier, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all parameters in the network initially
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the 'Second to Last' layer (Layer4)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Modify the 'Last' layer (The Classifier)
        num_features = self.resnet.fc.in_features
        
        # We output 1 value (logit). 
        # > 0 implies Class 1 (AI), < 0 implies Class 0 (Real)
        # We use a sequential block to add dropout for regularization 
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 1) 
        )

    def forward(self, x):
        return self.resnet(x)

def get_model():
    """
    Function to instantiate the model. 
    """
    model = AIOrNotClassifier()
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)  # should be [1, 1]


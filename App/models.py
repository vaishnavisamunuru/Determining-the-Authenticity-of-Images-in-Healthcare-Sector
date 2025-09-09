import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # compute output size after 3 pools for a 32×32 input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

cnn_model  = SimpleCNN().to(device)


resnet_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for name, param in resnet_backbone.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False

num_ftrs = resnet_backbone.fc.in_features
resnet_backbone.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

resnet_model = resnet_backbone.to(device)

densenet_backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
for name, param in densenet_backbone.named_parameters():
    if "features.norm5" not in name:
        param.requires_grad = False

densenet_backbone.features.conv0 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)
densenet_backbone.features.pool0 = nn.Identity()

num_ftrs = densenet_backbone.classifier.in_features
densenet_backbone.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)

densenet_model = densenet_backbone.to(device)

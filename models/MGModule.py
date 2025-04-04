import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

class BinaryClassificationHead(nn.Module):
    """
    This head predicts whether the sample belongs to class 0 or non-zero.
    It outputs 2 logits.
    """
    def __init__(self, input_dim, dropout_rate=0.5):
        super(BinaryClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out

class MultiClassificationHead(nn.Module):
    """
    This head predicts among the non-zero classes.
    Its output dimension is (num_bins - 1).
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(MultiClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out

class MGModule(nn.Module):
    def __init__(self, num_bins, pretrained_path=None):
        super(MGModule, self).__init__()
        # Load a pre-trained ResNet18 backbone.
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # The backbone's output feature dimension plus one for the continuous density.
        fused_dim = resnet.fc.in_features + 1
        
        # Head 1: Binary classification (0 vs. non-0)
        self.binary_head = BinaryClassificationHead(input_dim=fused_dim, dropout_rate=0.5)
        # Head 2: Multi-class classification for non–zero classes (classes 1 to num_bins–1)
        self.multi_head = MultiClassificationHead(input_dim=fused_dim, num_classes=num_bins - 1, dropout_rate=0.5)
        
        if pretrained_path is not None:
            self.mg_load_pretrained_model(pretrained_path)

    def mg_load_pretrained_model(self, mirai_path: str):
        mirai_weight = torch.load(mirai_path, map_location="cpu", weights_only=False)
        mirai_weight = mirai_weight.module._model
        model_modules = self.backbone._modules
        model_modules["0"].load_state_dict(mirai_weight.downsampler.conv1.state_dict())
        model_modules["1"].load_state_dict(mirai_weight.downsampler.bn1.state_dict())
        model_modules["2"].load_state_dict(mirai_weight.downsampler.relu.state_dict())
        model_modules["3"].load_state_dict(mirai_weight.downsampler.maxpool.state_dict())
        model_modules["4"]._modules["0"].load_state_dict(mirai_weight.layer1_0.state_dict())
        model_modules["4"]._modules["1"].load_state_dict(mirai_weight.layer1_1.state_dict())
        model_modules["5"]._modules["0"].load_state_dict(mirai_weight.layer2_0.state_dict())
        model_modules["5"]._modules["1"].load_state_dict(mirai_weight.layer2_1.state_dict())
        model_modules["6"]._modules["0"].load_state_dict(mirai_weight.layer3_0.state_dict())
        model_modules["6"]._modules["1"].load_state_dict(mirai_weight.layer3_1.state_dict())
        model_modules["7"]._modules["0"].load_state_dict(mirai_weight.layer4_0.state_dict())
        model_modules["7"]._modules["1"].load_state_dict(mirai_weight.layer4_1.state_dict())

    def forward(self, mg, density):
        # Extract features from the image using the backbone.
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1)
        # Ensure density has shape (batch_size, 1)
        if density.dim() == 1:
            density = density.unsqueeze(1)
        # Concatenate the image features with the continuous density value.
        fused_feature = torch.cat([mg_feature_flatten, density], dim=1)
        
        # First, predict using the binary head:
        #   - If the prediction is class 0 (index 0), then the sample is of class 0.
        #   - Otherwise, the sample is non–zero.
        binary_logits = self.binary_head(fused_feature)
        
        # Second, predict the specific non–zero class using the multi head.
        multi_logits = self.multi_head(fused_feature)
        
        return binary_logits, multi_logits, fused_feature
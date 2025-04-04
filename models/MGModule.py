import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class MGModule(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super(MGModule, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Replace the regression head with a classification head that outputs 40 logits.
        self.head = ClassificationHead(input_dim=resnet.fc.in_features, dropout_rate=0.5)
        if pretrained_path is not None:
            self.mg_load_pretrained_model(pretrained_path)

    # Load mirai model as pretrained model
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

    def forward(self, mg):
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1, -1)
        out = self.head(mg_feature_flatten)
        return out, mg_feature_flatten

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
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
        self.fc_out = nn.Linear(64, 40)
        
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

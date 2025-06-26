import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
import torch.nn.init as init

class ClassificationHead(nn.Module):
    """
    This head performs classification into num_classes categories.
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out
    
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

class MgmoduleSingleheadMammOnly(nn.Module):
    def __init__(self, num_bins, freeze_backbone, pretrained_path=None):
        super(MgmoduleSingleheadMammOnly, self).__init__()
        # Load a pre-trained ResNet18 backbone.
        resnet = torchvision.models.resnet18(weights=False)
        # Remove the final fully connected layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feature_dim = resnet.fc.in_features

        # Classification head unchanged
        self.classification_head = ClassificationHead(input_dim=512,
                                                      num_classes=num_bins,
                                                      dropout_rate=0.5)
        
        if pretrained_path is not None:
            self.mg_load_pretrained_model(pretrained_path)
        
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                print(name)
                # if name.startswith(("6", "7")):
                #     p.requires_grad = True 
                #     self.backbone.apply(init_weights)
                # else:
                #     p.requires_grad = False  # conv1, bn1, maxpool
                p.requires_grad = True
                self.backbone.apply(init_weights)

    def mg_load_pretrained_model(self, mirai_path: str):
        # Optionally load pretrained weights for the backbone.
        mirai_weight = torch.load(mirai_path, map_location="cpu", weights_only=False)
        mirai_weight = mirai_weight.module._model
        self.backbone._modules["0"].load_state_dict(mirai_weight.downsampler.conv1.state_dict())
        self.backbone._modules["1"].load_state_dict(mirai_weight.downsampler.bn1.state_dict())
        self.backbone._modules["2"].load_state_dict(mirai_weight.downsampler.relu.state_dict())
        self.backbone._modules["3"].load_state_dict(mirai_weight.downsampler.maxpool.state_dict())
        self.backbone._modules["4"]._modules["0"].load_state_dict(mirai_weight.layer1_0.state_dict())
        self.backbone._modules["4"]._modules["1"].load_state_dict(mirai_weight.layer1_1.state_dict())
        self.backbone._modules["5"]._modules["0"].load_state_dict(mirai_weight.layer2_0.state_dict())
        self.backbone._modules["5"]._modules["1"].load_state_dict(mirai_weight.layer2_1.state_dict())
        self.backbone._modules["6"]._modules["0"].load_state_dict(mirai_weight.layer3_0.state_dict())
        self.backbone._modules["6"]._modules["1"].load_state_dict(mirai_weight.layer3_1.state_dict())
        self.backbone._modules["7"]._modules["0"].load_state_dict(mirai_weight.layer4_0.state_dict())
        self.backbone._modules["7"]._modules["1"].load_state_dict(mirai_weight.layer4_1.state_dict())

    def forward(self, mg):
        # 1) image branch
        mg_feature = self.backbone(mg)
        mg_feat_flat = mg_feature.view(mg_feature.size(0), -1)  # [B, image_feature_dim]

        # 6) classification
        logits = self.classification_head(mg_feat_flat)                # [B,num_bins]
        return logits, mg_feat_flat


# Example usage:
if __name__ == "__main__":
    model = MgmoduleSingleheadMammOnly(num_bins=4)
    mg_input = torch.randn(8, 3, 224, 224)    # e.g., a batch of 8 images.
    logits, fused_features = model(mg_input)
    print("Logits shape:", logits.shape)
    print("Fused features shape:", fused_features.shape)

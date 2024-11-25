import torch.nn as nn
from torchvision.models import mobilenet_v2

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetBackbone, self).__init__()

        
        mobilenet = mobilenet_v2(pretrained=pretrained)

        
        self.stem = mobilenet.features[:3]  
        self.stage2 = mobilenet.features[3:7]   
        self.stage3 = mobilenet.features[7:14]  
        self.stage4 = mobilenet.features[14:]   

    def forward(self, x):
        #
        x = self.stem(x)
        stage2 = self.stage2(x) 
        stage3 = self.stage3(stage2)  
        stage4 = self.stage4(stage3)  

        #32， 96， 1280
        return stage2, stage3, stage4


    def freeze_bn(self):
        """
        Freeze all BatchNorm layers: set eval mode and disable gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # 设置为评估模式，冻结 running_mean 和 running_var
                for param in module.parameters():
                    param.requires_grad = False  # 冻结参数

    def freeze_stages(self, stage):

        if stage >= 0:
            # Freeze Stem
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        # Freeze the requested stages
        for i in range(2, 4):  # Stage starts from stage2
            layer = getattr(self, f'stage{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


if __name__ == '__main__':
    model = MobileNetBackbone()
    print(model)
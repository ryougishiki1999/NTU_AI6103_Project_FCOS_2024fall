import torch.nn as nn
import torch.nn.functional as F
import math
from config import DefaultConfig


class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        
        if DefaultConfig.choosen_backbone == DefaultConfig.Backbone.ResNet50:
            self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        elif DefaultConfig.choosen_backbone == DefaultConfig.Backbone.MobileNetV2:
            self.prj_5 = nn.Conv2d(1280, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(96, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(32, features, kernel_size=1)
        elif DefaultConfig.choosen_backbone == DefaultConfig.Backbone.VovNet27Slim:
            self.prj_5 = nn.Conv2d(512, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(384, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(256, features, kernel_size=1)
        elif DefaultConfig.choosen_backbone == DefaultConfig.Backbone.ResNet18:
            self.prj_5 = nn.Conv2d(512, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(256, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(128, features, kernel_size=1)
                     
        # self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        # self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        # self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        
        P4 = P4 + self.upsamplelike([P5,C4])
        P3 = P3 + self.upsamplelike([P4,C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        # print("[Test] Shape of C3: ", C3.shape)
        # # save C3 to a npy file
        # import numpy as np
        # np.save('/root/autodl-tmp/res/feature_maps/C4.npy', C3.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/C5.npy', C4.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/P3.npy', P3.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/P4.npy', P4.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/P5.npy', P5.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/P6.npy', P6.detach().cpu().numpy())
        # np.save('/root/autodl-tmp/res/feature_maps/P7.npy', P7.detach().cpu().numpy())
        # print("[Test] Shape of C4: ", C4.shape)
        # print("[Test] Shape of C5: ", C5.shape)
        # print("[Test] Shape of P3: ", P3.shape)
        # print("[Test] Shape of P4: ", P4.shape)
        
        # import numpy as np
        # import os
        
        # featuremap_dict = {
        #     'C3': C3,
        #     'C4': C4,
        #     'C5': C5,
        #     'P3': P3,
        #     'P4': P4,
        #     'P5': P5,
        #     'P6': P6,
        #     'P7': P7
        # }
        # for featuremap_name, featuremap in featuremap_dict.items():
        #     print(f"[Test] Shape of {featuremap_name}: ", featuremap.shape)
        #     np.save(os.path.join(DefaultConfig.out_featuremaps_dir_path, f"{featuremap_name}.npy"), featuremap.detach().cpu().numpy())
            
        return [P3,P4,P5,P6,P7]



import torch
import torch.nn as nn
from mobilenetv2.models.imagenet import mobilenetv2

class id_digital_tampering_model(nn.Module):
    def __init__(self, dropout_rate=0.2, width_mult=1.):
        super(id_digital_tampering_model, self).__init__()
        self.backbone = mobilenetv2(width_mult=width_mult)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if width_mult == 1.:
            self.backbone.load_state_dict(torch.load('mobilenetv2/pretrained/mobilenetv2_1.0-0c6065bc.pth', map_location=self.device))
        elif width_mult == 0.75:
            self.backbone.load_state_dict(torch.load('mobilenetv2/pretrained/mobilenetv2_0.75-dace9791.pth', map_location=self.device))
        elif width_mult == 0.5:
            self.backbone.load_state_dict(torch.load('mobilenetv2/pretrained/mobilenetv2_0.5-eaa6f9ad.pth', map_location=self.device))
        elif width_mult == 0.25:
            self.backbone.load_state_dict(torch.load('mobilenetv2/pretrained/mobilenetv2_0.25-b61d2159.pth', map_location=self.device))

        modules = list(self.backbone.children())[:-3]
        modules.append(nn.AdaptiveAvgPool2d((1,1)))
        self.backbone = nn.Sequential(*modules)
        #print(self.backbone)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
        self.classifier = nn.Sequential(
            nn.Linear(in_features = int(320 * width_mult), out_features = 2, bias = True)
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
    def forward(self, img):
        #img = torch.cat((full_img, crop_img), dim = 0)
        backbone_output = self.backbone(img)
        backbone_output = backbone_output.view(backbone_output.size(0), -1)
        features_drop = self.dropout(backbone_output)
        outputs = self.classifier(features_drop)
        return (backbone_output, outputs)

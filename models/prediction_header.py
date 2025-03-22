import torch
import torch.nn as nn


class Header(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.out_h = args.sensor_sizeW
        self.out_w = args.sensor_sizeH
        self.num_joints = args.num_joints


        self.proj_x = nn.Sequential(
            nn.Conv2d(126, 128, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.num_joints, 1),
            nn.AdaptiveAvgPool2d((self.out_h, 1))
        )


        self.proj_y = nn.Sequential(
            nn.Conv2d(126, 128, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.num_joints, 1),
            nn.AdaptiveAvgPool2d((self.out_w, 1))
        )


        self.confidence = nn.Sequential(
            nn.Conv2d(126, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_joints, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, C, N, _ = x.shape


        x_pred = self.proj_x(x)
        x_pred = x_pred.squeeze(-1)
        output_x = self.sigmoid(x_pred) * self.out_h

        y_pred = self.proj_y(x)
        y_pred = y_pred.squeeze(-1)
        output_y = self.sigmoid(y_pred) * self.out_w
        

        conf = self.confidence(x)
        conf = conf.squeeze(-1).mean(dim=-1)
        return output_x, output_y, conf

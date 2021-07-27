import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3), # kernel_size, num_filters(=out_channels), stride, padding
    "M", # "M" is simply maxpooling with stride 2x2 and kernel 2x2
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # tuples and then last integer represents number of repeats
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    """Making CNN Block with Conv + batchnorm + leakyrelu(activation)"""
    def __init__(self, in_channels, out_channels, **kwargs): # keyword arguments
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # batchnorm을 사용할 것이기 때문에 bias = False
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)


    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        # x = self.darknet(x)
        # return self.fcs(torch.flatten(x, start_dim=1)) # batch size를 살리기 위해서 start_dim = 1
        return self.fcs(self.darknet(x))


    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                )
                in_channels = x[1]

            elif type(x) == str:
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            
            elif type(x) == list:
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Integer

                for _ in range(num_repeats):
                    layers.extend([
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        ),

                        CNNBlock(
                            in_channels=conv1[1], # 두번째 conv layer의 input channel은 conv1의 output channel과 동일
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        ),
                    ])

                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(), # 첫번째 축의 정보는 살리고 flatten을 한다 (batch_size)
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)) # output = (N, S, S, 30)
        )


def test(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

if __name__ == "__main__":
    test()
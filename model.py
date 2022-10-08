import torch 
from torch import conv2d, nn
# SRCNN model

# in_channels, out_channesls, kernels and after is Relu
config_conv2d = [ 
    (3, 65, 5),
    (65, 12, 1),
    (12, 12, 3),
    (12, 12, 3),
    (12, 12, 3),
]

class SRCNN(nn.Module) :
    def __init__(self, in_channels = 3):
        super().__init__()

        self.conv2d_layers = self._conv2d()
        self.conv2dT = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=9, stride=2, padding=3)

    def _conv2d(self) :
        cov2d_layers = []
        for layer in config_conv2d :
            cov2d_layers.append( 
                nn.Conv2d(layer[0], layer[1], layer[2], padding='same')
            )
            cov2d_layers.append( 
                nn.ReLU()
            )
        
        cov2d_layers = nn.Sequential(*cov2d_layers)
        return cov2d_layers

    def forward(self, x) :
        x = self.conv2d_layers(x)
        x = self.conv2dT(x)
        return x

if __name__ == '__main__' :
    image = torch.randn(size=(8,3,64,64)) # B  C W H
    model = SRCNN(in_channels=3)
    x = model.forward(image)
    print(x.shape)
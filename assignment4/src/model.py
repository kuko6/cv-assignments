from torch import nn
from torchinfo import summary

def build_conv_block(in_channels, out_channels, double=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]

    # add another convolution
    if double:
        layers.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])

    # add downsampling
    layers.extend([
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
        #nn.MaxPool2d(kernel_size=2),
        nn.ReLU()
    ])

    return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], blocks=4, classes=29, double=False):
        super().__init__()
        self.features = nn.Sequential()

        for i in range(blocks):
            if i == (blocks-1):
                self.features.add_module(f'block{i}', build_conv_block(in_channels, channels[i], double=double))    
            else:
                self.features.add_module(f'block{i}', build_conv_block(in_channels, channels[i]))
            in_channels = channels[i]
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_channels*4*4, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.fc(out)
        
        return out   
    

if __name__ == '__main__':
    summary(Model(), (256, 3, 64, 64))
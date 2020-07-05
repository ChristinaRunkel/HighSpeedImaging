import torch
import torch.nn as nn


'''
Class to build one residual block with 5 Conv3d layers and leaky ReLUs in between
'''
class ResidualBlock(nn.Module):
    def __init__(self, dropout=False, dummy_dropout_layers=False):
        super().__init__()
        
        if dummy_dropout_layers:
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=1, kernel_size=(3,3,3), padding=(1,1,1))
            )
        elif dropout:
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.3, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.4, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.5, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Dropout3d(p=0.5, inplace=True),
                nn.Conv3d(in_channels=60, out_channels=1, kernel_size=(3,3,3), padding=(1,1,1))
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.LeakyReLU(),
                nn.Conv3d(in_channels=60, out_channels=1, kernel_size=(3,3,3), padding=(1,1,1))
            )
        
    def forward(self, x):
        identity = x
        x = self.layers(x)
        x += identity
        return x


class FinalBlock(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=num_blocks, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=60, out_channels=60, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=60, out_channels=1, kernel_size=(3,3,3), padding=(1,1,1))
        )
        
    def forward(self, x):
        return self.layers(x)


'''
Network which calculates the loss at the very end of all residual blocks
'''
class SingleLossNet(nn.Module):
    def __init__(self, num_residual_blocks=3):
        super().__init__()
        self.layers = nn.Sequential(*[ResidualBlock() for _ in range(num_residual_blocks)])
        
    def forward(self, x):
        return self.layers(x)


'''
Network that calculates the loss after each residual block
'''
class MultiLossNet(nn.Module):
    def __init__(self, num_residual_blocks=3, final_block=False, dropout=False, dummy_dropout_layers=False):
        super().__init__()
        total_blocks = num_residual_blocks+1 if final_block else num_residual_blocks
        self.blocks = [0] * total_blocks
        self.layers = nn.Sequential(*[ResidualBlock(dropout=dropout, dummy_dropout_layers=dummy_dropout_layers) for _ in range(num_residual_blocks)])
        self.final_block = FinalBlock(num_residual_blocks) if final_block else None
        
    def forward(self, x):
        for i in range(len(self.layers)):
            if i==0:
                self.blocks[i] = self.layers[i](x)
            else:
                # call block i with output from previous block
                self.blocks[i] = self.layers[i](self.blocks[i-1])
        if self.final_block is not None:
            self.blocks[-1] = self.final_block(torch.cat(self.blocks[:-1], dim=1))
        return self.blocks

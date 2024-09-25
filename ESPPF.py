class ESPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1,1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv3 = Conv(c_,c_,1,1)
        self.cv4 = Conv(c_*4,c_*3,1,1)
        self.cv5 = Conv(c1,c_,1,1)

        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x1 = self.cv5(x)
        x = self.cv1(x)
        x = self.cv3(x)
        x = torch.cat((x,self.m1(x),self.m1(self.m1(x)),self.m1(self.m1(self.m1(x)))),1)
        x = self.cv4(x)
        x = torch.cat((x1,x),1)
        return self.cv2(x)
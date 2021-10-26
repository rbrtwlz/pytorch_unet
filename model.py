import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
from itertools import chain

class UNet(nn.Module):
    def __init__(self, num_c=2, downsampling_channels=[64,128,256,512,1024]): 
        super().__init__()
        self.num_c, self.chs = num_c, downsampling_channels
        self.hook_cache = []
        self.contracting_net = self.contracting_path()
        self.middle = nn.Sequential(*self.doubleconv_upsample(self.chs[-2],self.chs[-1]))
        self.expansive_net = self.expansive_path()
        
    def forward(self, x):
        self.hook_cache = []
        x = self.contracting_net(x)
        x = self.middle(x) 
        return self.expansive_net(x)
    
    def hook_store(self, m, inp, outp):
        self.hook_cache.append(outp)
        
    def hook_concat(self, m, inp, outp):
        _,c,h,w = outp.shape
        for stored_t in self.hook_cache:
            if stored_t.shape[1]==c: 
                return torch.cat([center_crop(stored_t,(h,w)), outp], dim=1)
        
    def contracting_path(self):
        chs = [1] + self.chs[:-1] #[1,64,128,256,512]
        return nn.Sequential(*chain(*[self.doubleconv_pool(in_c=c, out_c=chs[i+1]) for i,c in enumerate(chs[:-1])]))

    def expansive_path(self):
        chs = list(reversed(self.chs)) #[1024,512,256,128,64]
        return nn.Sequential(*chain(*[self.doubleconv_upsample(in_c=c, out_c=chs[i+1], outp_layer=(i+2==len(chs))) 
                                          for i,c in enumerate(chs[:-1])]))
        
    def doubleconv_upsample(self, in_c, out_c, concat_outp=True, outp_layer=False):
        layers = self.doubleconv(in_c, out_c, store_outp=False)
        if not outp_layer:
            layers += [nn.ConvTranspose2d(out_c, out_c//2, kernel_size=(2,2), stride=2)]
        else:
            layers += [nn.Conv2d(self.chs[0],self.num_c, kernel_size=(1,1))]
        if concat_outp: layers[-1].register_forward_hook(self.hook_concat)
        return layers
        
    def doubleconv_pool(self, in_c, out_c):
        return self.doubleconv(in_c, out_c) + [nn.MaxPool2d(kernel_size=(2,2))]
    
    def doubleconv(self, in_c, out_c, store_outp=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=(3,3)), nn.ReLU(), 
                  nn.Conv2d(out_c, out_c, kernel_size=(3,3)), nn.ReLU()]
        if store_outp: layers[-1].register_forward_hook(self.hook_store)  
        return layers
    
    
'''
UNet(
  (contracting_net): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (6): ReLU()
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (8): ReLU()
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (11): ReLU()
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (13): ReLU()
    (14): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    (16): ReLU()
    (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
    (18): ReLU()
    (19): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (middle): Sequential(
    (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
  )
  (expansive_net): Sequential(
    (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    (5): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
    (6): ReLU()
    (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (8): ReLU()
    (9): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (10): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
    (11): ReLU()
    (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (13): ReLU()
    (14): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (15): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
    (16): ReLU()
    (17): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (18): ReLU()
    (19): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
  )
)
'''

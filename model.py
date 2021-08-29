import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

class UNet(nn.Module):
    def __init__(self, num_c=2, downsampling_channels=[1,64,128,256,512,1024]): 
        super().__init__()
        self.num_c, self.chs = num_c, downsampling_channels
        self.hook_cache = []
        self.contracting_net = self.contracting_path()
        self.middle = self.doubleconv_upsample(self.chs[-2],self.chs[-1])
        self.expansive_net = self.expansive_path()
        self.output_layer = self.double_conv(self.chs[2],self.chs[1], output_layer=True, store_outp=False)
        
    def forward(self, x):
        x = self.contracting_net(x)
        x = self.middle(x) 
        x = self.expansive_net(x)
        return self.output_layer(x)
    
    def hook_store(self, m, inp, outp):
        self.hook_cache.append(outp)
        
    def hook_concat(self, m, inp, outp):
        _,c,h,w = outp.shape
        for stored_t in self.hook_cache:
            if stored_t.shape[1]==c: 
                return torch.cat([center_crop(stored_t,(h,w)), outp], dim=1)
        
    def contracting_path(self):
        channels = self.chs[:-1] #[64,128,256,512]
        return nn.Sequential(*[self.doubleconv_pool(in_c=c, out_c=channels[i+1]) for i,c in enumerate(channels[:-1])])

    def expansive_path(self):
        channels = list(reversed(self.chs))[:-2] #[1024,512,256,128]
        return nn.Sequential(*[self.doubleconv_upsample(in_c=c, out_c=channels[i+1]) for i,c in enumerate(channels[:-1])])
        
    def doubleconv_upsample(self, in_c, out_c, concat_outp=True):
        layers = nn.Sequential(*[self.double_conv(in_c, out_c, store_outp=False), nn.ConvTranspose2d(out_c, out_c//2, kernel_size=(2,2), stride=2)])
        if concat_outp: layers.register_forward_hook(self.hook_concat)
        return layers
        
    def doubleconv_pool(self, in_c, out_c):
        return nn.Sequential(*[self.double_conv(in_c, out_c), nn.MaxPool2d(kernel_size=(2,2))])
    
    def double_conv(self, in_c, out_c, store_outp=True, output_layer=False):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=(3,3)), nn.ReLU(), 
                  nn.Conv2d(out_c, out_c, kernel_size=(3,3)), nn.ReLU()]
        if output_layer: layers += [nn.Conv2d(self.chs[1],self.num_c, kernel_size=(1,1))]
        layers = nn.Sequential(*layers)
        if store_outp: layers.register_forward_hook(self.hook_store)  
        return layers

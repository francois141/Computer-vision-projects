import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReLU(nn.Module):
    """My custom implementation of ConvReLU"""
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1,pad = 1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ConvBnReLU(nn.Module):
    """My custom implementation of ConvBnReLU"""
    def __init__(self,in_channels,out_channels,kernel_size = 3,stride = 1,pad = 1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # group 1
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # group 2
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # group 3
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # last convolution
        self.last_conv = nn.Conv2d(32,32,3,1,1)

    def forward(self, x):
        # simple forward pass
        tensor = x.float()
        tensor = self.conv3(self.conv2(self.conv1(self.conv0(tensor))))
        tensor = self.conv7(self.conv6(self.conv5(self.conv4(tensor))))
        return self.last_conv(tensor)


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # create all the blocks
        self.convrel1 = ConvReLU(G,8,stride=1)
        self.convrel2 = ConvReLU(8,16,stride=2)
        self.convrel3 = ConvReLU(16,32,stride=2)
        # create last conv layer
        self.conv = torch.nn.Conv2d(8,1,3,1,1,1)
        # create convtranspose layers
        self.convtranspose1 = torch.nn.ConvTranspose2d(in_channels=32,out_channels=16,stride=2,kernel_size=3,padding=1,output_padding=1)
        self.convtranspose2 = torch.nn.ConvTranspose2d(in_channels=16,out_channels=8, stride=2,kernel_size=3,padding=1,output_padding=1)


    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        B,G,D,H,W = x.size()

        x = x.transpose(1,2).reshape(B*D, G, H, W)

        c0 = self.convrel1(x)
        c1 = self.convrel2(c0)
        c2 = self.convrel3(c1)
        c3 = self.convtranspose1(c2)
        c4 = self.convtranspose2(c3 + c1)
        output = self.conv(c4 + c0).squeeze()

        return torch.reshape(output,(B,D,H,W))

def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device),])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # create points
        xyz = torch.stack((x, y, torch.ones_like(x)))  
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1).double()  
        # Apply rotation
        rot_xyz = torch.matmul(rot, xyz) 
        # Add depth to the rotation
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, D, 1) * depth_values.view(B, 1, D, 1)  
        # Apply translation
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  
        # Project points
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :] 
        # Normalize points 
        proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1  
        proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        # stack points
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  
     
    warped_src_fea = F.grid_sample(src_fea,grid.view(B, D * H, W, 2).float())

    return warped_src_fea.view(B,C, D, H,W)



def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    B,C,D,H,W = warped_src_fea.size()
    size = C // G
    # reshape ref_fea
    ref_fea = ref_fea.unsqueeze(2).expand(B,C,D,H,W)
    ref_fea= torch.reshape(ref_fea,(B,size,C//size,D,H,W))
    # reshape warped_src_fea
    warped_src_fea = torch.reshape(warped_src_fea,(B,size,C//size,D,H,W))
    # compute the sum
    return torch.sum(ref_fea * warped_src_fea * (G/C),dim=1)

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    depth_values = torch.reshape(depth_values,(p.shape[0],p.shape[1],1,1)).expand(p.shape[0],p.shape[1],p.shape[2],p.shape[3])
    return torch.sum(depth_values*p,dim=1)

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    loss = nn.L1Loss()
    return loss(depth_est * mask,depth_gt * mask)

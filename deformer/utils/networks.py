# From https://github.com/yifita/deep_cage
import torch.nn.functional as F
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.utils.spectral_norm as spectral_norm
from .graph_layers import GraphResBlock, GraphLinear
from .nn import Conv1d, Linear
# from deep_cage.pytorch_points.pytorch_points.network.pointnet2_modules import PointnetSAModuleMSG

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        # self.args = args
        self.emb_dims=512
        self.dp=False
        self.k = 10
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                #    self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                #    self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                #    self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                #    self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512,self.emb_dims, kernel_size=1, bias=False),
                                #    self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dp)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dp)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        x = F.leaky_relu((self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu((self.linear2(x)), negative_slope=0.2) 
        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        # # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        # x = self.dp2(x)
        # import ipdb
        # ipdb.set_trace()
        # x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x
    
class PointNetfeat_nopool(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        # x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std
class ElaIN(nn.Module):
    def __init__(self, norm_nc, addition_nc):
        super().__init__()
        
        self.mlp_same = nn.Conv1d(addition_nc, norm_nc, 1)
        self.mlp_gamma = nn.Conv1d(norm_nc, norm_nc, 1)
        self.mlp_beta = nn.Conv1d(norm_nc, norm_nc, 1)

        self.mlp_weight = nn.Conv1d(2*norm_nc, norm_nc, 1)

    def forward(self, x, addition):

        # feature dim align
        addition = self.mlp_same(addition)

        # get gamma and beta
        addition_gamma = self.mlp_gamma(addition)
        addition_beta = self.mlp_beta(addition)

        # calculate the mean of identity features and warped features in dim=2
        id_avg = torch.mean(addition, 2 ,keepdim=True)
        x_avg = torch.mean(x, 2, keepdim=True)
        
        # get the adaptive weight
        weight_cat = torch.cat((id_avg, x_avg), 1)
        weight = self.mlp_weight(weight_cat)
        
        # calculate the final modulation parameters
        x_mean, x_std = calc_mean_std(x)
        gamma = addition_gamma * weight + x_std * (1-weight)
        beta = addition_beta * weight + x_mean * (1-weight)
            
        # normalization and denormalization    
        x = (x - x_mean) / x_std
        out = x * (1 + gamma) + beta

        return out

class ElaINResnetBlock(nn.Module):
    def __init__(self, fin, fout, ic):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv1d(fin, fmiddle, kernel_size=1)
        self.conv_1 = nn.Conv1d(fmiddle, fout, kernel_size=1)
        self.conv_s = nn.Conv1d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = ElaIN(fin, ic)
        self.norm_1 = ElaIN(fmiddle, ic)
        self.norm_s = ElaIN(fin, ic)

    def forward(self, x, addition):
        x_s = self.conv_s(self.actvn(self.norm_s(x, addition)))
        dx = self.conv_0(self.actvn(self.norm_0(x, addition)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, addition)))
        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
class Tw_predictor(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        # print(opt.d_residual,'d_residual')
        # self.encoder = PointNetfeat(dim=3, num_points=6890, bottleneck_size=opt.bottleneck_size, normalization=opt.normalization).cuda()
        self.encoder2 = PointNetfeat_nopool(dim=3, num_points=6890, bottleneck_size=opt.bottleneck_size//4, normalization=opt.normalization).cuda()
        # self.nd_decoder = MLPDeformer3(dim=24,bottleneck_size=1024, npoint=6890, residual=False, normalization=opt.normalization).cuda()
        self.tw=nn.Sequential(
        PointNetfeat(dim=opt.dim, num_points=opt.num_point, bottleneck_size=opt.bottleneck_size),
        Linear(opt.bottleneck_size,opt.bottleneck_size, activation="lrelu", normalization=opt.normalization),
        MLPDeformer3(dim=opt.dim, bottleneck_size=opt.bottleneck_size, npoint=opt.n_keypoints*64,
                            residual=opt.d_residual, normalization=opt.normalization)).cuda()
        self.merger=nn.Sequential(Linear(opt.bottleneck_size//2,1, activation="lrelu", normalization=opt.normalization)).cuda()
    def forward(self,source_kp,source_shape):
        B, __,__ = source_shape.shape
        input_shapes = torch.cat([source_shape], dim=0)

        # distance=( source_shape.transpose(2,1)[:,:,None,:]-source_kp.transpose(2,1)[:,None,:,:])
        # # dis_norm=distance.pow(2) .sum(-1) 
        # dis_norm= (distance.pow(2).mul( input_precision[:,None,:,:]) ).sum(-1)
        
        # if self.opt.n_input==27:
        #     input_shapes = torch.cat([source_shape.transpose(2,1),dis_norm], dim=-1).transpose(2,1)
        
        # else:
        #      input_shapes = dis_norm.transpose(2,1)
        # import ipdb
        # ipdb.set_trace()
  
        # if self.opt.smal:
        #     input_shapes = torch.cat([source_shape.transpose(2,1),dis_norm], dim=-1).transpose(2,1).float()
        src_f=self.tw(source_shape) .reshape(B,24,-1)
        kp_f=self.encoder2(source_kp).permute(0,2,1)
        
        # shape_code =torch.max( self.encoder(input_shapes),dim=2)[0]  [:,:,None].repeat(1,1,kp_f.shape[2])

        # shape_code.unsqueeze_(-1)
        # s_code, t_code = torch.split(shape_code, B, dim=0)
        # target_code = torch.cat([s_code, t_code], dim=1)
        

        target_code = self.merger(torch.cat([src_f, kp_f], dim=2) )
        # skinning= self.nd_decoder(target_code)
        return target_code
    
class Skin_predictor(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        # print(opt.d_residual,'d_residual')
        self.encoder = PointNetfeat_nopool(dim=opt.n_input, num_points=6890, bottleneck_size=opt.bottleneck_size*4, normalization=opt.normalization).cuda()
        # self.nd_decoder = MLPDeformer3(dim=24,bottleneck_size=1024, npoint=6890, residual=False, normalization=opt.normalization).cuda()
        self.merger = nn.Sequential(
                Conv1d(opt.bottleneck_size*4,opt.bottleneck_size*2, 1, activation="lrelu", normalization=opt.normalization),
                Conv1d(opt.bottleneck_size*2,opt.num_joints, 1, activation="lrelu", normalization=opt.normalization),
            ).cuda()
    def forward(self,source_kp,source_shape,input_precision):
        B, __,__ = source_shape.shape
        # input_shapes = torch.cat([source_shape], dim=0)

        distance=( source_shape.transpose(2,1)[:,:,None,:]-source_kp.transpose(2,1)[:,None,:,:])
        # dis_norm=distance.pow(2) .sum(-1) 
        dis_norm= (distance.pow(2).mul( input_precision[:,None,:,:]) ).sum(-1)
        
        if self.opt.n_input==27:
            input_shapes = torch.cat([source_shape.transpose(2,1),dis_norm], dim=-1).transpose(2,1)
        
        else:
             input_shapes = dis_norm.transpose(2,1)
             
        # import ipdb
        # ipdb.set_trace()   
        if self.opt.smal:
            input_shapes = torch.cat([source_shape.transpose(2,1),dis_norm], dim=-1).transpose(2,1).float()
        shape_code = self.encoder(input_shapes)
        # shape_code.unsqueeze_(-1)
        # s_code, t_code = torch.split(shape_code, B, dim=0)
        # target_code = torch.cat([s_code, t_code], dim=1)
        target_code = (self.merger(shape_code).exp().transpose(2,1)).softmax(2)
        
        # import ipdb
        # ipdb.set_trace()
        # skinning= self.nd_decoder(target_code)
        return target_code
class ElaINGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = 16 #64
        self.fc = nn.Conv1d(3, 16 * nf, 3, padding=1)

        self.conv1 = torch.nn.Conv1d(16 * nf, 16 * nf, 1) 
        self.conv2 = torch.nn.Conv1d(16 * nf, 8 * nf, 1) 
        self.conv3 = torch.nn.Conv1d(8 * nf, 4 * nf, 1) 
        self.conv4 = torch.nn.Conv1d(4 * nf, 3, 1) 

        self.elain_block1 = ElaINResnetBlock(16 * nf, 16 * nf, 256)
        self.elain_block2 = ElaINResnetBlock(8 * nf, 8 * nf, 256)
        self.elain_block3 = ElaINResnetBlock(4 * nf, 4 * nf, 256)
        # self.fea=PointNet2feat(dim=3, num_points=6890, bottleneck_size=256, normalization=opt.normalization)
        self.fea=PointNetfeat_nopool(dim=3, num_points=6890, bottleneck_size=256, normalization=opt.normalization)
        # self.out=PointNetfeat_nopool(dim=3, num_points=6890, bottleneck_size=3, normalization=opt.normalization)
    def forward(self, identity_features, warp_out):
        x = warp_out.transpose(2,1)
        
        # import ipdb
        # ipdb.set_trace()
        
        addition = self.fea(identity_features)

        x = self.fc(x)
        x = self.conv1(x)
        x = self.elain_block1(x, addition)
        x = self.conv2(x)
        x = self.elain_block2(x, addition)
        x = self.conv3(x)
        x = self.elain_block3(x, addition)
        # import ipdb
        # ipdb.set_trace()        
        x = self.conv4(x)
        # print(x[0][:20])

        return x


class PointNetfeat(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class MLPDeformer3(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint)
            )
    def forward(self, code):
        B, _ = code.shape
        x = self.layers(code)
        x = x.reshape(B, self.npoint)
        # print(x.shape)
        
        return x
class MLPDeformer2(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint* dim*2 )
            )
    def forward(self, code):
        B, _ = code.shape
        x = self.layers(code)
        x = x.reshape(B, self.dim*2, self.npoint)
        # print(x.shape)
        
        return x


class PointNet2feat(nn.Module):
    """
    pointcloud (B,3,N)
    return (B,bottleneck_size)
    """
    def __init__(self, dim=3, num_points=2048, num_levels=3, bottleneck_size=512, normalization=None):
        super().__init__()
        assert(dim==3)
        self.SA_modules = nn.ModuleList()
        self.postSA_mlp = nn.ModuleList()
        NPOINTS = []
        RADIUS = []
        MLPS = []
        start_radius = 0.2
        start_mlp = 24
        self.l_output = []
        for i in range(num_levels):
            NPOINTS += [num_points//4]
            num_points = num_points//4
            RADIUS += [[start_radius, ]]
            start_radius *= 2
            final_mlp = min(256, start_mlp*4)
            MLPS += [[[start_mlp, start_mlp*2, final_mlp], ]]
            start_mlp *= 2
            self.l_output.append(start_mlp)

        bottleneck_size_per_SA = bottleneck_size // len(MLPS)
        self.bottleneck_size = bottleneck_size_per_SA*len(MLPS)

        in_channels = 0
        for k in range(len(MLPS)):
            mlps = [[in_channels]+mlp for mlp in MLPS[k]]
            in_channels = 0
            for idx in range(len(MLPS[k])):
                in_channels += MLPS[k][idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=[32,], mlps=mlps, normalization=normalization)
                )
            self.postSA_mlp.append(Conv1d(in_channels, bottleneck_size_per_SA, 1, normalization=normalization, activation="tanh"))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, return_all=False):
        pointcloud = pointcloud.transpose(1,2).contiguous()
        li_xyz, li_features = self._break_up_pc(pointcloud)

        # B,C,N
        # l_xyz, l_features = [xyz], [li_features]
        l_xyz, l_features = [], []
        for i in range(len(self.SA_modules)):
            # Pointnetmodule + MLP + maxpool
            li_xyz, li_features = self.SA_modules[i](li_xyz, li_features)
            li_features_post = self.postSA_mlp[i](li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features_post)

        # max pool (B,4*#SA,1) all SAmodules
        # exclude the first None features
        global_code = torch.cat([torch.max(l_feat, dim=-1)[0] for l_feat in l_features], dim=1)

        l_features.append(global_code)
        l_xyz.append(None)
        if return_all:
            return l_features, l_xyz
        else:
            return global_code

class MLPDeformer(nn.Module):
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint*dim)
            )
    def forward(self, code, template):
        B, C, N = template.shape
        
        assert(self.npoint == N)
        assert(self.dim == C)
        if code.ndim > 2:
            code = code.view(B, -1)
        x = self.layers(code)
        x = x.reshape(B,C,N)
        if self.residual:
            x += template
        return x

class MLPDeformer_large(nn.Module):
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 1024, activation="lrelu", normalization=normalization),
                Linear(1024, 2048, activation="lrelu", normalization=normalization),
                Linear(2048, npoint*dim)
            )
    def forward(self, code, template):
        B, C, N = template.shape
        
        assert(self.npoint == N)
        assert(self.dim == C)
        if code.ndim > 2:
            code = code.view(B, -1)
        x = self.layers(code)
        x = x.reshape(B,C,N)
        if self.residual:
            x += template
        return x

class Offset_predictor(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.encoder = PointNetfeat(dim=opt.dim, num_points=opt.num_point, bottleneck_size=opt.bottleneck_size, normalization=opt.normalization)
        self.nd_decoder = MLPDeformer(dim=opt.dim, bottleneck_size=opt.bottleneck_size*2, npoint=opt.n_keypoints,
                                        residual=False, normalization=opt.normalization)
        self.merger = nn.Sequential(
                Conv1d(opt.bottleneck_size*2,opt.bottleneck_size*2, 1, activation="lrelu", normalization=opt.normalization),
            )
    def forward(self,source_kp,source_shape,target_shape):
        B, __,__ = source_shape.shape
        input_shapes = torch.cat([source_shape, target_shape], dim=0)
        shape_code = self.encoder(input_shapes)
        shape_code.unsqueeze_(-1)
        s_code, t_code = torch.split(shape_code, B, dim=0)
        target_code = torch.cat([s_code, t_code], dim=1)
        target_code = self.merger(target_code)
        offset= self.nd_decoder(target_code, source_kp)
        return offset,target_code 
    
class GraphCNN(nn.Module):
        
    def __init__(self, num_layers=5, num_channels=256):
        super(GraphCNN, self).__init__()
        # self.A = A
        # self.ref_vertices = ref_vertices
        self.num_layers=num_layers
        # self.resnet = resnet50(pretrained=True)
        layers = nn.ModuleList()
        layers.append(GraphLinear(3 + 512, 2 * num_channels).cuda())
        layers.append(GraphResBlock(2 * num_channels, num_channels).cuda())
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels).cuda())
        self.shape=nn.ModuleList()
        self.shape.append( GraphResBlock(num_channels, 64).cuda())
        self.shape.append( GraphResBlock(64, 32).cuda())
        self.shape.append(   nn.GroupNorm(32 // 8, 32).cuda())
        self.shape.append(     nn.ReLU(inplace=True).cuda())
        self.shape.append(     GraphLinear(32, 3).cuda())
        self.layers=layers
        # self.gc = nn.Sequential(*layers)
        # self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                             nn.ReLU(inplace=True),
        #                             GraphLinear(num_channels, 1),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(A.shape[0], 3))

    def forward(self,ref_vertices,A,g):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        
        batch_size = ref_vertices.shape[0]
        # ref_vertices = ref_vertices[None, :, :].expand(batch_size, -1, -1)
        
        # image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices.transpose(2,1),g.expand(-1,-1,6890)], dim=1)
        x=self.layers[0](x)
        # import ipdb
        # ipdb.set_trace()
        for i in range(self.num_layers):
            x = self.layers[i+1](x,A)
        for i in range(len(self.shape)): 
            if i<2:
                x= self.shape[i](x,A)
            else:
                x = self.shape[i](x)
        return x

class Offset_predictor2(nn.Module):
    def __init__(self,opt):
        super().__init__()
        # print(opt.d_residual,'d_residual')
        self.encoder = PointNetfeat(dim=opt.dim, num_points=opt.num_point, bottleneck_size=opt.bottleneck_size*2, normalization=opt.normalization)
        self.nd_decoder = MLPDeformer_large(dim=opt.dim, bottleneck_size=opt.bottleneck_size*4, npoint=6890,
                                        residual=False, normalization=opt.normalization)
        self.merger = nn.Sequential(
                Conv1d(opt.bottleneck_size*4,opt.bottleneck_size*4, 1, activation="lrelu", normalization=opt.normalization),
            )
    def forward(self,source_kp,source_shape,target_shape):
        B, __,__ = source_shape.shape
        input_shapes = torch.cat([source_shape, target_shape], dim=0)
        shape_code = self.encoder(input_shapes)
        shape_code.unsqueeze_(-1)
        s_code, t_code = torch.split(shape_code, B, dim=0)
        target_code = torch.cat([s_code, t_code], dim=1)
        target_code = self.merger(target_code)
        offset= self.nd_decoder(target_code, source_kp)
        return offset,target_code
class PosEncoder():
    def __init__(self, number_frequencies, include_identity):
        freq_bands = torch.pow(2, torch.linspace(0., number_frequencies - 1, number_frequencies))
        self.embed_fns = []
        self.output_dim = 0
        self.number_frequencies = number_frequencies
        self.include_identity = include_identity
        if include_identity:
            self.embed_fns.append(lambda x: x)
            self.output_dim += 1
        if number_frequencies > 0:
            for freq in freq_bands:
                for periodic_fn in [torch.sin, torch.cos]:
                    self.embed_fns.append(lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq))
                    self.output_dim += 1

    def encode(self, coordinate):
        return torch.cat([fn(coordinate) for fn in self.embed_fns], -1)


class WeightPred(nn.Module):
    
    def __init__(self, opt_weight ):
        super(WeightPred, self).__init__()
        self.num_neuron = opt_weight['total_dim']
        self.num_layers = opt_weight['num_layers']
        self.num_parts = opt_weight['num_parts']

        self.shape = opt_weight['beta']
        self.body_enc = opt_weight['body_enc']
        self.input_dim = 3 + self.num_parts   # (X, theta)
        if self.body_enc:
            self.input_dim = 3 + self.num_parts    # (X- jts, theta)

        self.layers = nn.ModuleList()
        self.pose_enc = opt_weight['pose_enc']
        if self.shape:
            self.input_dim = self.input_dim +10
        ### apply positional encoding on input features
        if self.pose_enc:
            x_freq = opt_weight['x_freq']
            jts_freq = opt_weight['jts_freq']
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + self.num_parts * 2 * jts_freq # (X, PE(X), PE(theta))
            #todo: add for # (X, PE(X- jts), PE(theta)) and # ( PE(X- jts), PE(theta))
            if self.body_enc:
                self.input_dim = 24 + self.num_parts * 3 +72 * 2 * x_freq  #(theta, PE(X- jts))
                self.input_dim =  72 + 72 * 2 * x_freq  #(PE(|X- jts|))
                self.input_dim =  24 + 24 * 2 * x_freq  #(PE(|X- jts|))
                self.input_dim = 3 + self.num_parts + 3 * 2 * x_freq + self.num_parts * 2 * jts_freq  # (X, PE(X), PE(theta))

            self.x_enc = PosEncoder(x_freq, True)
            self.jts_enc = PosEncoder(jts_freq, True)

        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 24))
        self.actvn = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, jts,beta=None):
        batch_size = x.shape[0]
        num_pts = x.shape[1]

        if self.pose_enc:  #todo : check this
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)
            x = x.reshape(batch_size, num_pts, x.shape[1])

            jts = jts.reshape(jts.shape[0] * jts.shape[1], jts.shape[2])
            jts = self.jts_enc.encode(jts)
            jts = jts.reshape(batch_size, num_pts, jts.shape[1])
        for i in range(self.num_layers - 1):
            if i == 0:
                if self.shape:
                    x_net = torch.cat((x, jts, beta), dim=2)
                else:
                    x_net = torch.cat((x, jts), dim=2)
                #x_net = x
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net
        x_net = self.softmax(self.layers[-1](x_net))
        return x_net     


class Pointnet2(nn.Module):
    def __init__(self):
        super(Pointnet2, self).__init__()
        in_channel = 3
        # self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x=F.relu (self.fc1(x))
        x=F.relu (self.fc2(x))
        
        # import ipdb
        # ipdb.set_trace()
        # x = self.drop1(F.relu(self.fc1(x)))
        # x = self.drop2(F.relu(self.fc2(x)))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return 
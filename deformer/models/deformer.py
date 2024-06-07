from numpy import source
import pytorch3d.loss
import pytorch3d.utils
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
from  pytorch3d.structures.pointclouds import Pointclouds
import torch.utils.data
from einops import rearrange
from torch.optim import *
import torch.nn.functional as F
from  pytorch3d.structures.meshes import Meshes
# from pytorch_points.network.layers import Conv1d, Linear
# from arap import ARAP_Solver
# from ..pytorch_arap.arap import ARAP_from_meshes, add_one_ring_neighbours,add_n_ring_neighbours
# from ..pytorch_arap.arap import compute_energy as arap_loss
from ..utils.gmm import  deform_with_GMM,deform_with_GMM_key,get_weights
from deep_cage.pytorch_points.pytorch_points.network.operations import faiss_knn, dot_product, batch_svd, ball_query, group_knn
from ..utils.utils import resample_mesh, normalize_to_box
from ..utils.networks import Linear, MLPDeformer2, PointNetfeat,MLPDeformer,Offset_predictor,Offset_predictor2,ElaINGenerator,Skin_predictor,DGCNN,Pointnet2,MLPDeformer3,Tw_predictor
from deep_cage.pytorch_points.pytorch_points.network.geo_operations import batch_normals,compute_face_normals_and_areas
from ..utils.utils import normalize_to_box, sample_farthest_points
from .util import util
from ..meshlap.cot import MeshLaplacianLoss
from ..smpl_model.joint_reg import j_regressor,vertices2joints

from ..ik.lbs import batch_inverse_kinematics_transform, batch_inverse_kinematics_transform_naive,smal_batch_inverse_kinematics_transform_naive
import pickle as pk
import torch
import torch.nn as nn
from ..utils.networks import GraphCNN
# from ...utils.arap_interpolation import *
# from utils.interpolation_base import  *
# class ArapInterpolationEnergy(InterpolationEnergyHessian):
#     """The interpolation method based on Sorkine et al., 2007"""

#     def __init__(self):
#         super().__init__()

#     # override
#     def forward_single(self, vert_new, vert_ref, shape_i):
#         E_arap = arap_energy_exact(vert_new, vert_ref, shape_i.get_neigh())
#         return E_arap

#     # override
#     def get_hessian(self, shape_i):
#         return shape_i.get_neigh_hessian()
# from pytorch_points.network.operations import faiss_knn, dot_product, batch_svd, ball_query, group_knn
# from pytorch_points.utils.pytorch_utils import save_grad, linear_loss_weight
# from pytorch_points.network.model_loss import nndistance, labeled_nndistance
# from pytorch_points.network.geo_operations import (compute_face_normals_and_areas, dihedral_angle,
#                                                   CotLaplacian, UniformLaplacian, batch_normals)
# from pytorch_points.network.model_loss import (MeshLaplacianLoss, PointEdgeLengthLoss, \
                                            #    MeshStretchLoss, PointStretchLoss, PointLaplacianLoss,
                                            #    SimpleMeshRepulsionLoss, MeshEdgeLengthLoss,
                                            #    NormalLoss)
# from utils import adj2inc
# from pytorch_points.network.model_loss import (MeshLaplacianLoss, PointEdgeLengthLoss, \
#                                                MeshStretchLoss, PointStretchLoss, PointLaplacianLoss,
#                                                SimpleMeshRepulsionLoss, MeshEdgeLengthLoss,
#           
# NormalLoss)
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def compute_sum(a):
    sum__=torch.zeros_like(a)
    sum_=0
    for i in range(len(a)):
        sum_+=a[i]
        sum__[i]=sum_
    return sum__
smpl_path='./keypointdeformer/smpl_model/data/smpl/SMPL_NEUTRAL.pkl'
with open(smpl_path, 'rb') as smpl_file:
    smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))
    


NUM_BODY_JOINTS = 23
LEAF_NAMES = [
    'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
]
JOINT_NAMES = [
'pelvis', 'left_hip', 'right_hip',      # 2
'spine1', 'left_knee', 'right_knee',    # 5
'spine2', 'left_ankle', 'right_ankle',  # 8
'spine3', 'left_foot', 'right_foot',    # 11
'neck', 'left_collar', 'right_collar',  # 14
'jaw',                                  # 15
'left_shoulder', 'right_shoulder',      # 17
'left_elbow', 'right_elbow',            # 19
'left_wrist', 'right_wrist',            # 21
'left_thumb', 'right_thumb',            # 23
'head', 'left_middle', 'right_middle',  # 26
'left_bigtoe', 'right_bigtoe'           # 28
]
ROOT_IDX = JOINT_NAMES.index('pelvis')
LEAF_IDX = [JOINT_NAMES.index(name) for name in LEAF_NAMES]
SPINE3_IDX = 9
def _parents_to_children( parents):
    children = torch.ones_like(parents) * -1
    for i in range(24):
        if children[parents[i]] < 0:
            children[parents[i]] = i
    # for i in LEAF_IDX:
    #     if i < children.shape[0]:
    #         children[i] = -1

    # children[SPINE3_IDX] = -3
    children[0] = 3
    children[SPINE3_IDX] = JOINT_NAMES.index('neck')

    return children

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def central_distance_mean_score(points, gt_points, faces):
    score = 0
    # print(points.shape)
    # print(gt_points.shape)

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        connected_points_index = np.delete(np.unique(faces[connected_trianlges,:]), point_index)
        # print(connected_points_index)
        connected_points= points[:,connected_points_index]
        gt_connected_points= gt_points[:,connected_points_index]
        # import ipdb
        # ipdb.set_trace()
        current_point_array = points[:,point_index][None].repeat(connected_points.shape[1],1).transpose(1,0)
        gt_current_point_array = gt_points[:,point_index][None].repeat(connected_points.shape[1], 1).transpose(1,0)
        # print(current_point_array)

        distance = connected_points - current_point_array
        gt_distance = gt_connected_points - gt_current_point_array
        loss = nn.MSELoss()
        score += loss(distance, gt_distance)

    return torch.mean(score)

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)
def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]


    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms,transforms

# num_joints=24
NUM_JOINTS=23
parents = torch.zeros(len(JOINT_NAMES), dtype=torch.long)
parents[:(NUM_JOINTS + 1)] = to_tensor(to_np(smpl_data.kintree_table[0])).long()
parents[0] = -1
child=_parents_to_children(parents)

def remove_0(a):
        return a[0]
class NormalLoss(torch.nn.Module):
    """
    compare the PCA normals of two point clouds assuming known or given correspondence
    ===
    params:
        pred : (B,N,3)
        gt   : (B,N,3)
        idx12: (B,N)
    """
    def __init__(self, nn_size=10, reduction="mean"):
        super().__init__()
        self.nn_size = nn_size
        self.reduction = reduction
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, gt, pred, idx12=None):
        gt_normals, idx = batch_normals(gt, nn_size=self.nn_size, NCHW=False)
        if idx12 is not None:
            pred = torch.gather(pred, 1, idx12.unsqueeze(-1).expand(-1,-1,3))
            pred_normals, _ = batch_normals(pred, nn_size=self.nn_size, NCHW=False)
        else:
            pred_normals, _ = batch_normals(pred, nn_size=self.nn_size, NCHW=False, idx=idx)

        # compare the normal with the closest point
        loss = 1-self.cos(pred_normals, gt_normals)
        if self.reduction == "mean":
            return loss.mean(loss)
        elif self.reduction == "max":
            return (torch.max(loss, dim=-1)[0]).mean()
        elif self.reduction == "sum":
            return torch.sum(loss, dim=-1).mean()
        elif self.reduction == "none":
            return loss

class LocalFeatureLoss(torch.nn.Module):
    """
    penalize point to surface loss
    Given points (B,N,3)
    1. find KNN and the center
    2. fit PCA, get normal
    3. project p-center to normal
    """
    def __init__(self, nn_size=10, metric=torch.nn.MSELoss("mean"), **kwargs):
        super().__init__()
        self.nn_size = nn_size
        self.metric = metric

    def forward(self, xyz1, xyz2, **kwargs):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        B,N,C = xyz1.shape
        grouped_points, idx, _ = group_knn(self.nn_size, xyz1, xyz1, unique=True, NCHW=False)
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        # fit pca
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        # BN,C,k
        U, S, V = batch_svd(allpoints)
        # V is BNxCxC, last_u BNxC
        normals = V[:, :, -1].view(B, N, C).detach()
        # FIXME what about the sign of normal
        ptof1 = dot_product((xyz1 - group_center.squeeze(2)), normals, dim=-1)

        # for xyz2 use the same neighborhood
        grouped_points = torch.gather(xyz2.unsqueeze(1).expand(-1,N,-1,-1), 2, idx.unsqueeze(-1).expand(-1,-1,-1,C))
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        # MB,C,k
        U, S, V = batch_svd(allpoints)
        # V is MBxCxC, last_u MBxC
        normals = V[:, :, -1].view(B, N, C).detach()
        ptof2 = dot_product((xyz2 - group_center.squeeze(2)), normals, dim=-1)
        # compare ptof1 and ptof2 absolute value (absolute value can only determine bent, not direction of bent)
        loss = self.metric(ptof1.abs(), ptof2.abs())
        # # penalize flat->curve
        bent = ptof2-ptof1
        bent.masked_fill_(bent<0, 0.0)
        bent = self.metric(bent, torch.zeros_like(bent))
        # bent.masked_fill_(bent<=1.0, 0.0)
        loss += 5*bent
        return loss
def get_adjacency_matrix(edges):
    
        matrix = torch.zeros((edges.max()+1, edges.max()+1))
        matrix[edges[:,0], edges[:,1]] = 1

        return matrix
    
class Deformer(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        # parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--lambda_init_points", type=float, help="", default=2.0)
        parser.add_argument("--lambda_chamfer", type=float, help="", default=0.0)
        parser.add_argument("--lambda_l2_cycle", type=float, help="", default=1.0)
        # parser.add_argument("--lambda_edge_cycle", type=float, help="", default=0.0)
        parser.add_argument("--lambda_l2_pair", type=float, help="", default=1.0)
        parser.add_argument("--lambda_kp_cycle", type=float, help="", default=0.0)
        parser.add_argument("--lambda_sup", type=float, help="", default=0)
        parser.add_argument("--lambda_edge", type=float, help="", default=0)
        parser.add_argument("--lambda_edgec", type=float, help="", default=0)
        parser.add_argument("--lambda_edge_sup", type=float, help="", default=0)
        parser.add_argument("--lambda_edge_pair", type=float, help="", default=0)
        parser.add_argument("--lambda_edge_cycle", type=float, help="", default=0)
        parser.add_argument("--lambda_central", type=float, help="", default=0)
       
        parser.add_argument("--lambda_kp", type=float, help="", default=0.0)
        parser.add_argument("--lambda_lap1", type=float, help="", default=0)
        parser.add_argument("--lambda_lap2", type=float, help="", default=0)
        parser.add_argument("--lambda_normal", type=float, help="", default=0.0)
        parser.add_argument("--lambda_local", type=float, help="", default=0.0)
        parser.add_argument("--lambda_arap", type=float, help="", default=0.0)
        parser.add_argument("--decay", type=float, help="", default=0.1)
        parser.add_argument("--lambda_twsin", type=float, help="", default=0)
        parser.add_argument("--lambda_influence_predict_l2", type=float, help="", default=1e6)
        parser.add_argument("--iterations_init_points", type=float, help="", default=0)
        # parser.add_argument("--no_optimize_cage", action="store_true", help="")
        # parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        # parser.add_argument("--n_fps", type=int, help="")
        # parser.add_argument("--lambda_gcn",type=float, help="", default=0.1)
        
        parser.add_argument("--gl", type=str, help="", default='one')
        parser.add_argument("--n_input", type=int, help="", default=27)
        parser.add_argument("--tw", action="store_true", help="")
        parser.add_argument("--refine_like_smpl", action="store_true", help="")
        parser.add_argument("--d_residual", action="store_true", help="")
        parser.add_argument("--lable_detach",action="store_true", help="")
        parser.add_argument("--no_precision",action="store_true", help="")
        parser.add_argument("--input_precision",action="store_true", help="")
        parser.add_argument("--refine",action="store_true", help="")
        parser.add_argument("--smal",action="store_true", help="")
        
        # parser.add_argument("--no_precision",action="store_true", help="")
        return parser

    
    
    
    
    def __init__(self, opt):
        super(Deformer, self).__init__()
        
        self.opt = opt
        self.j_reg= j_regressor
        self.dim = self.opt.dim
        # self.interp_energy = ArapInterpolationEnergy()
        self.p2f_loss= LocalFeatureLoss(16, torch.nn.MSELoss(reduction="sum"))
        self.shape_normal_loss = NormalLoss(reduction="none", nn_size=16)
        # self.p2f_loss= LocalFeatureLoss(16, torch.nn.MSELoss(reduction="sum"))
        self.shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(reduction="mean"), use_cot=True,
                                                     use_norm=True, consistent_topology=False, precompute_L=False)
        # template_vertices, template_faces = self.create_cage()
        # self.init_template(template_vertices, template_faces)
        # self.p2f_loss = LocalFeatureLoss(16, torch.nn.MSELoss(reduction="none"))
        # self.shape_laplacian = PointLaplacianLoss(16, torch.nn.MSELoss(reduction="none"))
        self.init_networks(opt.bottleneck_size, self.opt.dim, opt)
        self.init_optimizer()
        if self.opt.smal:
            self.NUM_JOINTS = 32
        else:
            self.NUM_JOINTS = 23
    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.opt.ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.opt.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    

    def init_networks(self, bottleneck_size, dim, opt):
        # self.gcn=GraphCNN().cuda()
        if opt.smal:
            opt.num_joints=32
        else:
            opt.num_joints=23
        if opt.refine:
            self.offset_predictor2=ElaINGenerator(opt).cuda()
        self.skin_predictor=Skin_predictor(opt).cuda()
        # self.offset_predictor=nn.DataParallel(self.offset_predictor)
        # self.offset_predictor=self.offset_predictor.cuda()
        # keypoint predictor
        # print(opt.num_point)
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization))
        nd_decoder_kpt = MLPDeformer2(dim=dim, bottleneck_size=bottleneck_size, npoint=opt.n_keypoints,
                                residual=opt.d_residual, normalization=opt.normalization)
        self.keypoint_predictor = nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)
        if self.opt.tw==True:
            # self.tw=nn.Sequential(
            # PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size),
            # Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization),
            # MLPDeformer3(dim=dim, bottleneck_size=bottleneck_size, npoint=opt.n_keypoints,
            #                     residual=opt.d_residual, normalization=opt.normalization)).cuda()
            self.tw=Tw_predictor(opt)
        # self.fea=DGCNN()
        # self.fea= nn.Sequential(
        # DGCNN(),
        #     Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization))
        # self.keypoint_predictor = nn.Sequential(self.fea, nd_decoder_kpt)
        # self.keypoint_predictor=nn.DataParallel(self.keypoint_predictor, device_ids=opt.device_ids)
        self.keypoint_predictor=self.keypoint_predictor.cuda()
        # influence predictor
        # influence_size = self.opt.n_keypoints * self.template_vertices.shape[2]
        # shape_encoder_influence = nn.Sequential(
        #     PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=influence_size),
        #     Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization))
        # dencoder_influence = nn.Sequential(
        #         Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization),
        #         Linear(influence_size, influence_size, activation=None, normalization=None))
        # self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)
        

    # def init_template(self, template_vertices, template_faces):
    #     # save template as buffer
    #     self.register_buffer("template_faces", template_faces)
    #     self.register_buffer("template_vertices", template_vertices)
        
    #     # n_keypoints x number of vertices
    #     # print(template_vertices.shape,self.opt.n_keypoints)
    #     self.influence_param = nn.Parameter(torch.zeros(self.opt.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)


    def init_optimizer(self):
        # params = [{"params": self.gmm_predictor.parameters()}]
        # self.gmm_optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        # self.optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.opt.lr})
        if self.opt.tw==True:
            params = [{"params": self.tw.parameters()}]
            self.tw_optimizer = torch.optim.AdamW(params, lr=self.opt.lr)
            # self.scheduler4 =lr_scheduler.CosineAnnealingWarmRestarts(self.tw_optimizer, T_0=50, eta_min=1e-5)
            self.scheduler4 = lr_scheduler.MultiStepLR(self.tw_optimizer,milestones=[12500,25000,40000,50000], gamma=self.opt.decay)
        params = [{"params": self.keypoint_predictor.parameters()}]
        self.keypoint_optimizer = torch.optim.AdamW(params, lr=self.opt.lr)
        params = [{"params": self.skin_predictor.parameters()}]
        self.skin_optimizer = torch.optim.AdamW(params, lr=self.opt.lr)
        if self.opt.refine:
            params = [{"params": self.offset_predictor2.parameters()}]
            self.offset2_optimizer = torch.optim.AdamW(params, lr=self.opt.lr)
            self.scheduler3 = lr_scheduler.MultiStepLR(self.offset2_optimizer,milestones=[12500,25000,40000,50000], gamma=self.opt.decay)
            # self.scheduler3=lr_scheduler.CosineAnnealingWarmRestarts(self.offset2_optimizer, T_0=50, eta_min=1e-5)
        # self.scheduler = lr_scheduler.ExponentialLR(self.keypoint_optimizer, gamma=0.9)
        # self.scheduler2 = lr_scheduler.ExponentialLR(self.skin_optimizer, gamma=0.9)
        # self.scheduler3 = lr_scheduler.ExponentialLR(self.offset2_optimizer, gamma=0.9)
        self.scheduler = lr_scheduler.MultiStepLR(self.keypoint_optimizer,milestones=[12500,25000,40000,50000], gamma=self.opt.decay)
        # self.scheduler =lr_scheduler.CosineAnnealingWarmRestarts(self.keypoint_optimizer, T_0=50, eta_min=1e-5)
        self.scheduler2 = lr_scheduler.MultiStepLR(self.skin_optimizer ,milestones=[12500,25000,40000,50000], gamma=self.opt.decay)
        # self.scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(self.skin_optimizer, T_0=50, eta_min=1e-5)
        

    # def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
    #     """
    #     pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
    #     """
    #     for _ in range(iters):
    #         vector = -cage
    #         current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
    #         min_distance, _ = torch.min(current_distance, dim=2)
    #         do_update = min_distance > distance
    #         cage = cage + step * vector * do_update[:, None]
    #     return cage

    
    def forward(self,tsource_shape,source_f,target_f, source_shape, target_shape):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        self.device=source_shape.device
        B, _, _ = source_shape.shape
        # self.source_mesh=Shape()
        # self.original_v=original_v
        # self.priginal_f=original_f
        # import ipdb
        # ipdb.set_trace()

        
        self.target_shape = target_shape
        self.source_shape = source_shape
        self.source_f=source_f
        # import ipdb
        # ipdb.set_trace()
        # if self.opt.phase=='train':
       
        if self.opt.smal:
            import pickle
            smal_path='./keypointdeformer/smal_CVPR2017.pkl'
            with open(smal_path, 'rb') as smal_file:
                data_struct = Struct(**pickle.load(smal_file,
                                                encoding='latin1'))
                j_regressor = to_tensor(to_np(data_struct.J_regressor)).double()
        else:
            j_regressor  = self.j_reg
        # else:
        # import ipdb
        # ipdb.set_trace()

        #j_regressor
        if self.opt.smal:
            tsource_init_keypoints=vertices2joints(j_regressor.cuda(),tsource_shape.permute(0,2,1).double() ).permute(0,2,1)
            source_init_keypoints=vertices2joints(j_regressor.cuda(),source_shape.permute(0,2,1).double() ).permute(0,2,1)
            target_init_keypoints=vertices2joints(j_regressor.cuda(),target_shape.permute(0,2,1).double() ).permute(0,2,1)
        else:
            tsource_init_keypoints=vertices2joints(j_regressor.cuda(),tsource_shape.permute(0,2,1)).permute(0,2,1)
            source_init_keypoints=vertices2joints(j_regressor.cuda(),source_shape.permute(0,2,1) ).permute(0,2,1)
            target_init_keypoints=vertices2joints(j_regressor.cuda(),target_shape.permute(0,2,1) ).permute(0,2,1)
        self.init_keypoints=torch.cat(( source_init_keypoints,target_init_keypoints),dim=0 ).float()

        # self.source_v=source_v
        source_shape=source_shape.float()
        target_shape= target_shape.float()
        target_shape=target_shape.float()
        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape
        keypoints_ = self.keypoint_predictor(shape)
        # import ipdb 
        # ipdb.set_trace()

        keypoints=keypoints_[:,:3,:]
        precision_all=keypoints_[:,3:,:]

        sig=torch.nn.Sigmoid()
        precision=sig(precision_all[:B])
        if self.opt.input_precision:
            input_precision=precision
        else:
            input_precision=torch.ones_like(precision)
        if self.opt.no_precision:
            precision=torch.ones_like(precision)
        # keypoints = self.keypoint_predictor(shape)
        # keypoints=keypoints_[:,:3,:]
        # precision_all=keypoints_[:,3:,:]
      
        # ipdb.set_trace()
        # sig=torch.nn.Sigmoid()
        # precision=sig(precision_all[:B])
        # keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints

        self.shape = shape
        self.keypoints = keypoints
       
        if self.opt.tw:
            # tw = torch.tanh(self.tw(self.keypoints ,shape).squeeze() )

            # source_tw_sin, target_tw_sin = torch.split(tw, B, dim=0)
            source_tw_sin=torch.tanh(self.tw(  source_keypoints ,source_shape).squeeze() )
            target_tw_sin=torch.tanh(self.tw(  target_keypoints ,target_shape).squeeze() )
            source_tw_cos=torch.sqrt(1-source_tw_sin.pow(2))
            target_tw_cos=torch.sqrt(1-target_tw_sin.pow(2))
            tw_sin=source_tw_cos*target_tw_sin-target_tw_cos*source_tw_sin[None]
            tw_cos=target_tw_cos*source_tw_cos-target_tw_sin*source_tw_sin[None]
            # import ipdb 
            # ipdb.set_trace()
            tw_combine=torch.cat((tw_cos[:,:,None],tw_sin[:,:,None]),dim=-1)
            self.tw_sin=tw_sin
        # n_fps = self.opt.n_fps if self.opt.n_fps else 2 * self.opt.n_keypoints
        # n_fps=self.opt.n_keypoints
        # self.init_keypoints = sample_farthest_points(shape, n_fps)

        # if target_shape is not None:
        #     source_keypoints, target_keypoints = torch.split(self.init_keypoints, B, dim=0)
        # else:
        #     source_init_keypoints = self.init_keypoints
        #     target_init_keypoints = None

        # cage = self.template_vertices
        # if not self.opt.no_optimize_cage:
        #     cage = self.optimize_cage(cage, source_shape)
        if self.opt.phase=='train': 
            outputs = {
                "source_keypoints": source_keypoints,
                "target_keypoints": target_keypoints,
                'source_init_keypoints': source_init_keypoints,
                'target_init_keypoints': target_init_keypoints
            }
        else:
            outputs = {
                "source_keypoints": source_keypoints,
                "target_keypoints": target_keypoints
                # 'source_init_keypoints': source_init_keypoints,
                # 'target_init_keypoints': target_init_keypoints
            }

        # self.influence = self.influence_param[None]
        # self.influence_offset = self.influence_predictor(source_shape)

        # self.influence_offset = rearrange(
        #     self.influence_offset, 'b (k c) -> b k c', k=self.influence.shape[1], c=self.influence.shape[2])
        # self.influence = self.influence + self.influence_offset
        
        # distance = torch.sum((source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        # n_influence = int((distance.shape[2] / distance.shape[1]) * self.opt.n_influence_ratio)
        # n_influence = max(5, n_influence)
        # threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        # threshold = threshold[..., None]
        # keep = distance <= threshold
        # influence = self.influence * keep

        # base_cage = cage
        if self.opt.smal:
            parents=torch.tensor([-1,          0,          1,          2,          3,
                    4,          5,          6,          7,          8,
                    9,          6,         11,         12,         13,
                    6,         15,          0,         17,         18,
                    19,          0,         21,         22,         23,
                    0,         25,         26,         27,         28,
                    29,         30,         16]).cuda()    
            child=torch.tensor([1,2,3,4,5,6,15,8,9,10,-1,12,13,14,-1,16,32,18,19,20,-1,22,23,24,-1,26,27,28,29,30,-1,-1,-1 ]).cuda() 
        else:
            parents=torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21]).cuda()        
            child=torch.tensor([3,  4,  5,  6,  7,  8,  9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19,
            20, 21, 22, 23, -1, -1]).cuda()        
        # parents=parents[:24]
        # child=child[:24]
        kps=source_keypoints.permute(0,2,1)
        kpt=target_keypoints.permute(0,2,1)
        # import ipdb
        # ipdb.set_trace()
        

        if self.opt.smal:
            if self.opt.tw:
                rot_mats, rotate_rest_pose = smal_batch_inverse_kinematics_transform_naive(
                pose_skeleton=kpt,global_orient=self.opt.gl,parents=parents,children=child,rest_pose=kps,phis= tw_combine)
            else:    
                rot_mats, rotate_rest_pose = smal_batch_inverse_kinematics_transform_naive(
                pose_skeleton=kpt,global_orient=self.opt.gl,parents=parents,children=child,rest_pose=kps,phis= 'zero')
        else:
            if self.opt.tw:
                rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_naive(
                pose_skeleton=kpt,global_orient=self.opt.gl,parents=parents,children=child,rest_pose=kps,phis= tw_combine)
            else:
                rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_naive(
                pose_skeleton=kpt,global_orient=self.opt.gl,parents=parents,children=child,rest_pose=kps,phis='zero')

        J_transformed, A,G = batch_rigid_transform(rot_mats,kps, parents)
        self.deformed_k=J_transformed
        
        # source_patches=torch.zeros(B,3,6890,3).cuda()
        # for ind in range( B):
        #     ind_diff= torch.where(source_f[ind][:,:,None]==torch.arange(6890)[None,None,:].cuda())
        #     reordered,order=ind_diff[2].sort()
        #     counts=torch.unique(reordered,return_counts=True)[1]
        #     index=compute_sum(counts)-1
        #     shape_ind=source_f[ind] [ ind_diff[0][order][index]]
        #     shape_ind[:,:,None]==torch.arange(6890)[None,None,:].cuda()
            
        #     source_patches[ind]=source_shape[ind][:,shape_ind] 
            
            
        # skinning =self.skin_predictor( kps.permute(0,2,1) ,source_shape ,source_patches.reshape(B,9,6890))
        # skinning =self.skin_predictor(kps.permute(0,2,1) ,source_shape)
        skinning =self.skin_predictor(kps.permute(0,2,1) ,source_shape,input_precision[:,:,:].transpose(1, 2))
        self.skinning=skinning        
        
        
        
        # keypoints_offset,g_code =self.offset_predictor(source_keypoints ,source_shape,target_shape)
        # cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        # new_cage = base_cage + cage_offset

        # cage = cage.transpose(1, 2)
        # new_cage = new_cage.transpose(1, 2)
        # precision_part=torch.zeros(B,9,self.opt.n_keypoints).cuda()

        tkps_start=tsource_init_keypoints.permute(0,2,1) [:,parents[1:]]
        middle_pts=(tkps_start+tsource_init_keypoints.permute(0,2,1)[:,1:])/2
        # precision=precision.view(B,self.opt.n_keypoints,3)
        if self.opt.lable_detach is True:
            weights=get_weights(precision[:,:,:self.NUM_JOINTS].transpose(1, 2),
                middle_pts,tsource_shape.transpose(1, 2)).detach()
        else:
            weights=get_weights(precision[:,:,:self.NUM_JOINTS].transpose(1, 2),
                middle_pts,tsource_shape.transpose(1, 2))     
        # precision=torch.ones(B,1,23).cuda()
        # weights=get_weights(precision[:,:,:23].transpose(1, 2),
        # middle_pts,source_shape.transpose(1, 2))
        
        # # ipdb.set_trace()
        self.weights=weights.float()

        T_weighted=torch.matmul(skinning,A[:,1:].reshape(B,self.NUM_JOINTS,-1)).reshape(B,-1,4,4)
        # import ipdb
        # ipdb.set_trace()
        num_verts=source_shape.shape[2]
        source_shape_homo=torch.cat([ source_shape.transpose(1, 2),torch.ones(B,num_verts,1 ).cuda()],dim=-1)
        deformed_c=torch.matmul(T_weighted,source_shape_homo.reshape(B,-1,4,1))[:,:,:3,0]
        self.deformed_c=deformed_c
        
        if self.opt.refine:
            refinement=self.offset_predictor2(source_shape, deformed_c).transpose(2,1)

            if self.opt.refine_like_smpl:
                source_shape_homo_refine=torch.cat([ source_shape.transpose(1, 2)+refinement,torch.ones(B,-1,1 ).cuda()],dim=-1)
                self.deformed_v=torch.matmul(T_weighted,source_shape_homo_refine.reshape(B,-1,4,1))[:,:,:3,0]
            else:
                self.deformed_v=refinement+deformed_c
        # print(precision)
        else:
            self.deformed_v=deformed_c
        # deformed_v, weights, _ = deform_with_GMM(precision,
        #     source_keypoints.transpose(1, 2),keypoints_offset.transpose(1, 2), source_v, verbose=True)
        # print(precision[2][1],weights[1].max())
        B=self.source_shape.shape[0]
        # import ipdb
        # ipdb.set_trace()
        # lap= torch.mean(self.shape_laplacian(self.source_shape.permute(0,2,1), self.deformed_v,self.source_f[0][None] ).view(B,-1))
        # print('cotlap:',lap)
        # adjmat = A[0].todense()
        # incmat_np, signmat_np, _ = adj2inc(adjmat)
        # incmat = torch.from_numpy(incmat_np).to_sparse().to(device)
        # signmat = torch.from_numpy(signmat_np).to_sparse().to(device)
        # arap_solver = ARAP_Solver(adjmat, incmat, signmat, device)
    
        
        # with torch.no_grad():
        #      deformed_shapes_asap = arap_solver(source_shape, deformed_shapes)
        
        self.target_v=target_shape
        
        # points=resample_mesh(deformed_m,2048)

        # deformed_c,weights_,__= deform_with_GMM_key(precision,
        #     source_keypoints.transpose(1, 2),keypoints_offset.transpose(1, 2), source_shape.transpose(1, 2), verbose=True)
        # import ipdb
        self.target_keypoints=target_keypoints
        # self.deformed_keypoints=source_keypoints+keypoints_offset
        # import ipdb
        # ipdb.set_trace()
        # A=torch.stack(list(map(get_adjacency_matrix, source_f)))
        # self.deformed_v =deformed_s
        self.deformed_c=deformed_c
        # de_deformed_c=deformed_c.detach()
        
        # refinement,g_code2 =self.offset_predictor2(de_deformed_c.transpose(2,1) ,de_deformed_c.transpose(2,1),source_shape)
        # self.deformed_v=self.offset_predictor2(source_shape,deformed_c).transpose(2,1)+deformed_c
        # refinement=self.opt.lambda_gcn* self.gcn(deformed_s,A.cuda(),g_code)
        # import ipdb
        # ipdb.set_trace()
        # self.deformed_v = rearrange(de_deformed_c.transpose(1, 2) + refinement,'b d n -> b n d' )
        # print(weights_[1].max())
        # ipdb.set_trace()
        outputs.update({
            # "cage": cage,
            # "cage_face": self.template_faces,
            # "new_cage": new_cage,
            "source_keypoints_deformed":self.deformed_k,
            # "deformed_s":deformed_s,
            "deformed_c": self.deformed_c ,
            "deformed": self.deformed_v ,
            "source":source_shape.permute(0,2,1),
            "target":target_shape.permute(0,2,1),
            "source_face":source_f,
            "target_face":target_f,
          
            # "influence": influence,
            "weights": skinning,
            })
        
        return outputs
            
    
    def compute_loss(self, iteration):
        losses = {}
        # import ipdb
        # ipdb.set_trace()
        if self.opt.lambda_init_points > 0:
            mse=torch.nn.MSELoss()
            # import ipdb
            # ipdb.set_trace()
            init_points_loss = mse(rearrange(self.keypoints, 'b d n -> b n d'), 
                rearrange(self.init_keypoints, 'b d n -> b n d'))
            losses['init_points'] = self.opt.lambda_init_points * init_points_loss
            l2_kp =  mse(self.skinning,self.weights)
            # l2_kp =  mse(self.target_k,self.deformed_k)
            losses['kp'] = self.opt.lambda_kp * l2_kp
        # E_x_0 = self.interp_energy.forward_single(
        #     self.source_shape  , self.deformed_v, self.source_mesh
        # ) + self.interp_energy.forward_single(
        #     self.deformed_v, self.source_shape, self.source_mesh
        # )
        
                # losses["p2f"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.p2f_weight, self.opt.p2f_weight/10)

        # ,
        #         dim=-1, keepdim=True)
        if iteration > self.opt.iterations_init_points:
            if self.opt.lambda_local > 0:
                local_loss=LocalFeatureLoss( 16, torch.nn.MSELoss(reduction="none"))
                losses['local']= torch.mean(local_loss( self.deformed_v,
                    rearrange(self.source_shape, 'b d n -> b n d')))*self.opt.lambda_local  
                # losses['localc']= torch.mean(local_loss( self.deformed_c,
                #     rearrange(self.source_shape, 'b d n -> b n d')))*self.opt.lambda_local  
            if self.opt.lambda_normal> 0:
                losses['normal']=   ((torch.mean(
                    self.shape_normal_loss( self.deformed_v,rearrange(self.source_shape, 'b d n -> b n d') ), dim=-1, keepdim=True)).mean() )* self.opt.lambda_normal
                           
            # if self.opt.lambda_kp> 0:
            #     l2_kp =  mse(self.skinning,self.weights)
            #     # l2_kp =  mse(self.target_k,self.deformed_k)
            #     losses['kp'] = self.opt.lambda_kp * l2_kp

            if self.opt.lambda_twsin> 0:
                losses['twsin']=torch.mean(abs(self.tw_sin))*self.opt.lambda_twsin
            if self.opt.lambda_edge> 0:
                losses['edge']=0
                losses['edgec']=0
                for i in range(len(self.source_shape)):  
                    f = self.source_f[i].cpu().numpy()
                    v = self.source_shape[i].transpose(0,1).cpu().numpy()
                    # import ipdb
                    # ipdb.set_trace()
                    face_lossc=util.compute_score(self.deformed_c[i].transpose(0,1),f,util.get_target(v,f,1))
                    weights=face_lossc.detach()
                    # import ipdb
                    # ipdb.set_trace()
                    
                    # face_mid=(self.source_shape[i].permute(1,0)[self.source_f[i][:,0]]+self.source_shape[i].permute(1,0)[self.source_f[i][:,1]]+self.source_shape[i].permute(1,0)[self.source_f[i][:,2]])/3
                    # dis=torch.norm(face_mid[:,None,:]-self.source_k[i][None],dim=-1)
                    # dis_min=torch.min(dis,dim=-1)
                    # mask=torch.where(dis_min[0]>0.1,dis_min[0],torch.zeros_like(dis_min[0]).cuda())
                    # de_mask=mask.detach()
                    # mask=mask/de_mask.max()
                    
                    face_loss= weights*util.compute_score(self.deformed_v[i].transpose(0,1),f,util.get_target(v,f,1))
                    
                    losses['edgec'] = losses['edgec'] + face_lossc.mean()
                    losses['edge'] = losses['edge'] + face_loss.mean()
                     
                losses['edgec'] = losses['edgec']/len(self.source_shape) * self.opt.lambda_edgec
                losses['edge'] = losses['edge']/len(self.source_shape) * self.opt.lambda_edge
            # if self.opt.lambda_arap > 0:
            #     # import ipdb 
            #     # ipdb.set_trace()
            #     self.source_m=Meshes(  self.source_shape.permute(0,2,1), self.source_f[0].repeat(self.source_shape.shape[0],1,1) )
            #     # losses['arap']=0
            #     # for i in range(self.source_shape.shape[0]):
            #     self.arapmeshes = ARAP_from_meshes(self.source_m, device=self.device)
            #     # import ipdb
            #     # ipdb.set_trace()
            #     losses['arap']=arap_loss(self.arapmeshes, self.source_shape.permute(0,2,1), self.deformed_v, device=self.device)*self.opt.lambda_arap
            #     # losses['arap']=losses['arap']/self.source_shape.shape[0]
            # import ipdb
            # ipdb.set_trace()
            if self.opt.lambda_chamfer > 0:
                chamfer_loss_c = pytorch3d.loss.chamfer_distance(
                    self.deformed_c,
                    rearrange(self.target_shape, 'b d n -> b n d'))[0]
                
                # chamfer_loss = pytorch3d.loss.chamfer_distance(
                #     self.deformed_v,
                #     rearrange(self.target_shape, 'b d n -> b n d'))[0]
                    # Pointclouds(self.deformed_v).points_padded(), Pointclouds(self.target_v).points_padded(),Pointclouds(self.deformed_v).num_points_per_cloud(),Pointclouds(self.target_v).num_points_per_cloud() )[0]
                losses['chamfer'] = self.opt.lambda_chamfer *( chamfer_loss_c)
            # if self.opt.lambda_lap2 > 0:
            #     # losses['lapl']=0 
            #     # losses["p2f"]=0
            #     # for i in range(len(self.source_shape)):
            #         # print(i)
            #     # import ipdb
            #     # ipdb.set_trace()
            #         # losses["p2f"] += self.p2f_loss(self.source_v[i][None,:,:],self.deformed_v[i][None,:,:])
            #     from ..utils.lapl import mesh_laplacian
            #     # import ipdb 
            #     # ipdb.set_trace()
            #     list_source_v= list(map( remove_0,list(torch.chunk(self.source_shape.permute(0,2,1),B ))  ))
            #     s_meshes=Meshes(list_source_v,self.source_f )
            #     list_deformed_v= list(map( remove_0,list(torch.chunk(self.deformed_v,B ))  ))
            #     deformed_meshes=Meshes(list_deformed_v,self.source_f )
            #     lap1=mesh_laplacian( s_meshes,"cot" )
            #     lap2=mesh_laplacian( deformed_meshes,"cot")
            #     mse=torch.nn.MSELoss(reduction='sum')
            #     # import ipdb
            #     # ipdb.set_trace()
            #     losses['lapl']=mse(lap1,lap2)*B*self.opt.lambda_lapl
                # lap1=self.shape_laplacian( self.source_shape.permute(0,2,1) ,self.deformed_v , face=self.source_f[0].repeat(self.source_shape.shape[0],1,1) )
                # lap2=self.shape_laplacian( self.source_shape.permute(0,2,1) ,self.deformed_c , face=self.source_f[0].repeat(self.source_shape.shape[0],1,1) )
                # losses['lap']=self.opt.lambda_lap2 *lap2
                    # print(losses['lapl'])
                # losses['lapl']=losses['lapl']/B*self.opt.lambda_lapl
                # losses['p2f']=losses['p2f']/B*self.opt.lamda_p2f

                # torch.mean( self.shape_laplacian( self.source_v,self.deformed_v, face=self.source_f).view(B,-1),dim=-1, keepdim=True)
            # if self.opt.lambda_influence_predict_l2 > 0:
            #     losses['influence_predict_l2'] = self.opt.lambda_influence_predict_l2 * torch.mean(self.influence_offset ** 2)

        return losses
    def compute_loss_sup(self,losses,gt,output):
        if self.opt.lambda_sup> 0:
            mse_mean=torch.nn.MSELoss()
            # import ipdb
            # ipdb.set_trace()
            sup_loss=mse_mean(output['deformed'],rearrange(gt, 'b d n -> b n d') )
            losses['sup']=self.opt.lambda_sup*sup_loss
        if self.opt.lambda_edge_sup> 0:
            losses['sup_edge']=0
            
            for i in range(len(self.source_shape)):  
                f = self.source_f[i].cpu().numpy()
                v = self.source_shape[i].transpose(0,1).cpu().numpy()
                # import ipdb
                # ipdb.set_trace()
                losses['sup_edge'] = losses['sup_edge'] + util.compute_score(output['deformed'][i].transpose(0,1),f,util.get_target(v,f,1))
            losses['sup_edge'] = losses['sup_edge']/len(self.source_shape) * self.opt.lambda_edge
            
    def compute_loss_cycle(self, losses,original_v,original_f,output,target_gt,t):
        if t > self.opt.iterations_init_points:

            if self.opt.lambda_central> 0:
                central_distance_loss= 0
                for i in range(original_v.shape[0]):
                    f = original_f[i].cpu().numpy()
                    # print(f.shape)#(13776, 3)
                    v = original_v[i]
                    # print(v.shape)#(1,6890, 3)
                    central_distance_loss += central_distance_mean_score(output['deformed'][i].transpose(0,1),v,f)
                losses['central_cy']=central_distance_loss/ (original_v.shape[0])*self.opt.lambda_central
            if self.opt.lambda_kp_cycle> 0: 
                mse_mean=torch.nn.MSELoss()
                mse_loss=mse_mean(output['target_keypoints'],rearrange(target_gt, 'b d n -> b n d'))
                losses['kp_cycle'] = self.opt.lambda_kp_cycle * mse_loss
            if self.opt.lambda_l2_cycle> 0: 
                mse_mean=torch.nn.MSELoss()
                mse_loss=mse_mean(output['deformed'],rearrange(original_v, 'b d n -> b n d'))
                losses['l2_cycle'] = self.opt.lambda_l2_cycle * mse_loss
                if self.opt.lambda_edge_cycle> 0: 

                    losses['edge_cycle']=0
                    for i in range(original_v.shape[0]):  
                        f = original_f[i].cpu().numpy()
                        v = original_v[i].transpose(0,1).cpu().numpy()
                        face_loss=util.compute_score(output['deformed'][i].transpose(0,1),f,util.get_target(v,f,1))
                        losses['edge_cycle'] = losses['edge_cycle'] + face_loss.mean()
                    losses['edge_cycle'] = losses['edge_cycle']/ (original_v.shape[0]) * self.opt.lambda_edge_cycle
                # import ipdb
                # ipdb.set_trace()                
        # if self.opt.lambda_l2_cycle> 0: 
        #     chamfer_loss = pytorch3d.loss.chamfer_distance(
        #         output['deformed'],
        #         rearrange(original_v, 'b d n -> b n d'))[0]

        #     # chamfer_loss = pytorch3d.loss.chamfer_distance(
        #     #     Pointclouds(output['deformed']).points_padded(), Pointclouds(original_v).points_padded(),Pointclouds(output['deformed']).num_points_per_cloud(),Pointclouds(original_v).num_points_per_cloud() )[0]
        #     losses['chamfer_cycle'] = self.opt.lambda_chamfer_cycle * chamfer_loss

        # if self.opt.lambda_influence_predict_l2 > 0:
        #     losses['influence_predict_l2'] = self.opt.lambda_influence_predict_l2 * torch.mean(self.influence_offset ** 2)

        # return losses
    def compute_loss_pair(self, losses,gt,gt_f,output,t):
        if t >self.opt.iterations_init_points:   
            if self.opt.lambda_l2_pair>0:  
                # mse=torch.nn.MSELoss(reduction='mean')
                # losses['l2_cycle_c'] = mse(rearrange(original_v, 'b d n -> b n d'),output['deformed_c'])
            # if t >10000:    
                mse=torch.nn.MSELoss(reduction='mean')
                losses['l2_pair'] = mse(rearrange(gt, 'b d n -> b n d'),output['deformed'])
                losses['l2_pair'] = self.opt.lambda_l2_pair* losses['l2_pair'] 

                if self.opt.lambda_central> 0:
                    central_distance_loss= 0
                    for i in range(gt.shape[0]):
                        f = gt_f[i].cpu().numpy()
                        # print(f.shape)#(13776, 3)
                        v = gt[i]
                        # print(v.shape)#(1,6890, 3)
                        central_distance_loss += central_distance_mean_score(output['deformed'][i].transpose(0,1),v,f)
                    losses['central_pair']=central_distance_loss/ (gt.shape[0])*self.opt.lambda_central  
                if self.opt.lambda_edge_pair> 0: 
                    # import ipdb
                    # ipdb.set_trace()

                    losses['edge_pair']=0
                    for i in range(gt.shape[0]):  
                        f = gt_f[i].cpu().numpy()
                        v = gt[i].transpose(0,1).cpu().numpy()
                        face_loss=util.compute_score(output['deformed'][i].transpose(0,1),f,util.get_target(v,f,1))
                        losses['edge_pair'] = losses['edge_pair'] + face_loss.mean()
                    losses['edge_pair'] = losses['edge_pair']/ (gt.shape[0]) * self.opt.lambda_edge_pair
    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)
        

    def optimize(self, losses, iteration):
        self.keypoint_optimizer.zero_grad()
        self.skin_optimizer.zero_grad()
        if self.opt.refine:
            self.offset2_optimizer.zero_grad()
        if self.opt.tw==True:
            self.tw_optimizer.zero_grad()
        if iteration <= self.opt.iterations_init_points: 
            keypoints_loss = self._sum_losses(losses, ['init_points','kp'])
            keypoints_loss.backward(retain_graph=True)
            self.keypoint_optimizer.step()
            self.skin_optimizer.step()
        if iteration > self.opt.iterations_init_points:

            loss = self._sum_losses(losses, ['init_points','l2_cycle','kp','sup','sup_edge','local','normal','edge','edgec','kp_cycle','l2_pair','edge_pair','edge_cycle','central_cy','central_pair','twsin'])

            loss.backward()
            # import ipdb
            # ipdb.set_trace()
            self.skin_optimizer.step()
            
            self.keypoint_optimizer.step()
            self.scheduler.step()
            self.scheduler2.step()
            
            if self.opt.refine==True:
                self.offset2_optimizer.step()
                self.scheduler3.step()
            if self.opt.tw==True:
                self.tw_optimizer.step()
                self.scheduler4.step()
    # def optimize_cycle(self, losses, iteration):
        # self.keypoint_optimizer.zero_grad()
        # self.offset_optimizer.zero_grad()

        # if iteration < self.opt.iterations_init_points: 
        #     keypoints_loss = self._sum_losses(losses, ['init_points'])
        #     keypoints_loss.backward(retain_graph=True)
        #     self.keypoint_optimizer.step()

        
            # self.keypoint_optimizer.step()
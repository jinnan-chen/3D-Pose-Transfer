import torch
import pytorch3d.io
from pytorch_points.network.operations import faiss_knn, dot_product, batch_svd, ball_query, group_knn
from pytorch_points.utils.pytorch_utils import save_grad, linear_loss_weight
from pytorch_points.network.model_loss import nndistance, labeled_nndistance
from pytorch_points.network.geo_operations import (compute_face_normals_and_areas, dihedral_angle,
                                                  CotLaplacian, UniformLaplacian, batch_normals)
from pytorch_points.network.model_loss import (MeshLaplacianLoss, PointEdgeLengthLoss, \
                                               MeshStretchLoss, PointStretchLoss, PointLaplacianLoss,
                                               SimpleMeshRepulsionLoss, MeshEdgeLengthLoss,
                                               NormalLoss)

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

shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(), use_cot=True, use_norm=True, consistent_topology=True, precompute_L=True)
from  pytorch3d.structures.meshes import  Meshes
mesh1=pytorch3d.io.load_objs_as_meshes(['/home/jin/Downloads/mixamo_dataset_obj/Claire/Crouch Cover To Standing Ready (2)/Crouch Cover To Standing Ready (2)_000002.obj'])
mesh2=pytorch3d.io.load_objs_as_meshes(['/home/jin/Downloads/mixamo_dataset_obj/Claire/Crouch Cover To Standing Ready (2)/Crouch Cover To Standing Ready (2)_000003.obj'])
mesh3=pytorch3d.io.load_objs_as_meshes(['/home/jin/Downloads/mixamo_dataset_obj/Dreyar/Dodging To The Right In Place (3)/Dodging To The Right In Place (3)_000005.obj'])
mesh4=pytorch3d.io.load_objs_as_meshes(['/home/jin/Downloads/mixamo_dataset_obj/Dreyar/Dodging To The Right In Place (3)/Dodging To The Right In Place (3)_000008.obj'])
f1=mesh1.faces_padded()
f2=mesh2.faces_padded()
f3=mesh3.faces_padded()
f4=mesh4.faces_padded()
v1=mesh1.verts_padded()
v2=mesh2.verts_padded()
v3=mesh3.verts_padded()
v4=mesh4.verts_padded()
# M1=Meshes([v1[0],v3[0]],[f1[0],f3[0]])
# M2=Meshes([v2[0],v4[0]],[f2[0],f4[0]])
# print(v1,v2.shape,f1.shape,f2.shape)
re=shape_laplacian(v1,v2,f1)
p2f_loss = LocalFeatureLoss(16, torch.nn.MSELoss(reduction="sum"))
re=p2f_loss(v1, v2)
print(re)
# re=shape_laplacian(M1.vert,M2.verts_padded(),M1.faces_padded())

# import math
import pytorch3d.io
import torch
import os 
from smpl_model.joint_reg import j_regressor,vertices2joints
def gmm_coordinates_3D_batch(query,kp,ctl):
    # print(precision)
    distance=(query[:,:,None,:]-kp[:,None,:,:])
    # import ipdb
    # ipdb.set_trace()
    # precision=precision[:,None,:,:,:]
    # sig=torch.nn.Sigmoid()
    # import ipdb
    # ipdb.set_trace()
    
    dis_norm= (- (   (distance.pow(2).mul(ctl)) .sum(-1) )).exp()
    # dis_= (distance[:,:,:,None,:].matmul( precision.repeat(1,query.shape[1],1,1,1))).matmul(distance[:,:,:,:,None])
    # print('dis',dis_.max(),dis_.min(),dis_norm_.max(),dis_norm_.min(),precision[1][0][1] )
    # dis_norm= torch.exp( -1/2* ( (distance[:,:,:,None,:].matmul(distance[:,:,:,:,None]) )))
    weights = (8* dis_norm).softmax(2)
    print(weights.max())
    # dis_norm= 1/ (1e-7+ (distance[:,:,:,None,:].matmul( precision)).matmul(distance[:,:,:,:,None]) )
    # weights = ( dis_norm.squeeze()).softmax(2)# h,j,n,1
    return weights

def deform_with_GMM_key(precision, kp, kp_deformation, query, verbose=False):
    """
    kp (B,N,3)
    kp_deformed (B,N,3)
    query (B,Q,3)
    """
    # weights_list= gmm_coordinates_3D(query, kp)
    # print(weights.shape,kp_deformation.shape)
    

    weights = gmm_coordinates_3D_batch(query, kp,precision)
    print(weights[0][1])
    deformed=query+weights.matmul(kp_deformation)
    if verbose:
        return deformed, weights, kp_deformation
    return deformed
def gmm_coordinates_3D(precision,query, kp):
    """
    params:
        query_list (N1,N2...NB,3)
        kp (B,N,3)
    return:
        weights  (N1,N2...NB, N)
    """
    # GMM
    # print(query.shape,kp.shape)
    # distance=(query[:,:,None,:]-kp[:,None,:,:])
    # dis_norm=torch.exp(-0.5* distance[:,:,:,None,:].matmul(distance[:,:,:,:,None]))
    kp_list=list( torch.chunk(kp,kp.shape[0],dim=0)  )
    precision=list(torch.chunk(precision,kp.shape[0],dim=0  ))
    # dis_norm = (self.ctl_ts.view(opts.n_hypo,-1,1,3) - pred_v.view(2*local_batch_size,opts.n_hypo,-1,3)[0,:,None].detach()) # p-v, H,J,1,3 - H,1,N,3
    # dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(self.ctl_rs).view(opts.n_hypo,-1,3,3)) # h,j,n,3
    # dis_norm = self.log_ctl.exp().view(opts.n_hypo,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
    weights_list=list(map(weights_comp,query,kp_list,precision))
    # weights = (-10 * dis_norm.sum(3)).softmax(2)[:,:,:,None] # h,j,n,1

    return weights_list
def deform_linear(query,kp_deformation,weights):
    "all lsit input"
    return weights.matmul(kp_deformation[0])+query

def deform_with_GMM(precision,kp, kp_deformation, query, verbose=False):
    """
    kp (B,N,3)
    kp_deformed (B,N,3)
    query (B,Q,3)
    """
    weights_list= gmm_coordinates_3D(precision,query, kp)
    # print(weights.shape,kp_deformation.shape)
    # weights = weights.detach()


    deformation_list = list( torch.chunk(kp_deformation,kp_deformation.shape[0],dim=0))
    
    # print(deformation.shape,query.shape)
    deformed=list(map(deform_linear, query,deformation_list,weights_list))
    if verbose:
        return deformed, weights_list, kp_deformation
    return deformed
from typing import Optional

def _save_mesh(f, verts, faces, decimal_places: Optional[int] = None) -> None:
    """
    Faster version of https://pytorch3d.readthedocs.io/en/stable/_modules/pytorch3d/io/obj_io.html

    Adding .detach().numpy() to the input tensors makes it 10x faster
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    verts, faces = verts.cpu().detach().numpy(), faces.cpu().detach().numpy()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)
            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)


def save_mesh(f, verts, faces, decimal_places: Optional[int] = None):
    with open(f, 'w') as f: 
        _save_mesh(f, verts, faces, decimal_places)

path_source='/ssd/jnchen/npt-data/id5_397.obj'
path_target='/ssd/jnchen/npt-data/id2_120.obj'
path_pair='/ssd/jnchen/npt-data/id2_20.obj'
path_gt='/ssd/jnchen/npt-data/id5_120.obj'
source_face= pytorch3d.io.load_objs_as_meshes([path_source], load_textures=False)[0].faces_padded()[0].cuda()
source_mesh= pytorch3d.io.load_objs_as_meshes([path_source], load_textures=False)[0].verts_padded()[0].unsqueeze(0).cuda()
target_mesh= pytorch3d.io.load_objs_as_meshes([path_target], load_textures=False)[0].verts_padded()[0].unsqueeze(0).cuda()
gt_mesh= pytorch3d.io.load_objs_as_meshes([path_gt], load_textures=False)[0].verts_padded()[0].unsqueeze(0).cuda()
pair_mesh= pytorch3d.io.load_objs_as_meshes([path_pair], load_textures=False)[0].verts_padded()[0].unsqueeze(0).cuda()
# print(source_mesh.shape)
# print(j_regressor.cuda().shape)
source_keypoints=vertices2joints(j_regressor.cuda(),source_mesh)
target_keypoints=vertices2joints(j_regressor.cuda(),target_mesh)
gt_keypoints=vertices2joints(j_regressor.cuda(),gt_mesh)
precision=torch.ones(1)+torch.randn(1,24,3)/3
precision=precision.cuda()
# print(gt_keypoints-source_keypoints)
deformed=deform_with_GMM_key(precision,source_keypoints,gt_keypoints-source_keypoints,source_mesh)
print(deformed.shape)
save_mesh( os.path.join('/ssd/jnchen', 'deformed_mesh.obj'), deformed[0], source_face)
save_mesh( os.path.join('/ssd/jnchen', 'gt_mesh.obj'), gt_mesh[0], source_face)
save_mesh( os.path.join('/ssd/jnchen', 's_mesh.obj'), source_mesh[0], source_face)
save_mesh( os.path.join('/ssd/jnchen', 't_mesh.obj'), target_mesh[0], source_face)
save_mesh( os.path.join('/ssd/jnchen', 'pair_mesh.obj'), pair_mesh[0], source_face)
# save_mesh( os.path.join('/ssd/jnchen', 'deformed_mesh.obj'), deformed, source_face)

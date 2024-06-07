# From https://github.com/yifita/deep_cage and https://github.com/yifita/pytorch_points
import math

import torch

def softmax(input, t=1.0):
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=2)
    return ex / sum[:,:,None]

def gmm_coordinates_3D_batch(query,kp,ctl):
    # print(precision)
    distance=(query[:,:,None,:]-kp[:,None,:,:])
    # import ipdb
    # ipdb.set_trace()
    # precision=precision[:,None,:,:,:]
    # sig=torch.nn.Sigmoid()
    # import ipdb
    # ipdb.set_trace()
    temp=ctl.mean(axis=2)
    dis_norm=(-100 * temp[:,None,:]* (distance.pow(2).mul( ctl[:,None,:,:]) ) .sum(-1) ).exp()
    weights = (10* dis_norm).softmax(2)
    # dis_norm= (- (   (distance.pow(2)) .sum(-1) )).exp()
    # dis_= (distance[:,:,:,None,:].matmul( precision.repeat(1,query.shape[1],1,1,1))).matmul(distance[:,:,:,:,None])
    # print('dis',dis_.max(),dis_.min(),dis_norm_.max(),dis_norm_.min(),precision[1][0][1] )
    # dis_norm= torch.exp( -1/2* ( (distance[:,:,:,None,:].matmul(distance[:,:,:,:,None]) )))
    # weights = (100* dis_norm).softmax(2)
    # print(weights.max())
    # dis_norm= 1/ (1e-7+ (distance[:,:,:,None,:].matmul( precision)).matmul(distance[:,:,:,:,None]) )
    # weights = ( dis_norm.squeeze()).softmax(2)# h,j,n,1
    return weights
def get_weights(precision, kp, query, verbose=False):
    """
    kp (B,N,3)
    kp_deformed (B,N,3)
    query (B,Q,3)
    """
    # weights_list= gmm_coordinates_3D(query, kp)
    # print(weights.shape,kp_deformation.shape)
    

    weights = gmm_coordinates_3D_batch(query, kp,precision)
    # deformed=query+weights.matmul(kp_deformation)
    # if verbose:
    #     return deformed, weights, kp_deformation
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



class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None

_scatter_add = ScatterAdd.apply

def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)


def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)


def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())

def weights_comp(query,kp,clt):
    kp=kp[0]
    distance=(query[:,None,:]-kp[None,:,:])
    # print(precision.shape)
    # print('ctl',clt.shape)
    # sig=torch.nn.Sigmoid()
    # dis_norm= sig  ((1/ ( 1e-7+ distance[:,:,None,:].matmul(precision.repeat(query.shape[0],1,1,1)  ).matmul(distance[:,:,:,None]))).squeeze())
    # dis_norm=(torch.exp (-1/2* (  distance[:,:,None,:].matmul(precision.repeat(query.shape[0],1,1,1)  ).matmul(distance[:,:,:,None])))).squeeze()
    dis_norm= clt.exp()* distance.pow(2)
    weights =(-5* dis_norm.sum(-1)) .softmax(1)
    # print(weights.max(),weights.min())
    
    return weights





def mean_value_coordinates_3D(query, vertices, faces, verbose=False, check_values=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    if check_values:
        assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    if check_values:
        assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    if check_values:
        assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (math.pi-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised

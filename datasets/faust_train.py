import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
# from utils.lbs import lbs
from ..utils_gcn.geometry import get_tpl_edges, fps_np, get_normal, calc_surface_geodesic
from ..utils_gcn.o3d_wrapper import Mesh, MeshO3d
import pickle
smpl_path='./keypointdeformer/smpl_model/data/smpl/SMPL_NEUTRAL.pkl'
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)
with open(smpl_path, 'rb') as smpl_file:
    data_struct = Struct(**pickle.load(smpl_file,
                                    encoding='latin1'))

j_regressor = to_tensor(to_np(data_struct.J_regressor))


class Faust_train(Dataset):
    def __init__(self, data_dir, flag=None, preload=True):
        super(Faust_train, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        if isinstance(flag, list) or isinstance(flag, tuple):
            self.names = flag
        elif flag is None:
            self.names = [k.replace('.obj', '') for k in os.listdir(os.path.join(data_dir)) if k.endswith('.obj')]
        # else:
        #     with open(os.path.join(self.data_dir, flag+'.txt')) as f:
        #         self.names = f.read().splitlines()

        self.vs, self.fs = [], []
        self.tpl_edge_indexs = []
        if self.preload:
            self._preload()
        else:
            raise ValueError

        print('Number of subjects:', len(self))

    def get(self, index):
        v, f, tpl_edge_index, name = self.load(index)
        tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
        tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=v.shape[0])

        # center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
        center=torch.mm(j_regressor,v)[0]
        scale=1
        # scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
        v0 = (v - center) / scale

        v0 = torch.from_numpy(v0).float()
        normal_v0 = get_normal(v0, f)
        return Data(v0=v0, tpl_edge_index=tpl_edge_index, triangle=f[None].astype(int),
                    feat0=normal_v0,
                    name=name, num_nodes=len(v0))

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.tpl_edge_indexs[index], self.names[index]

    def len(self):
        return len(self.names)

    def _preload(self):
        for idx in self.names:
            mesh_path = os.path.join(self.data_dir, idx+'.obj')
            m = Mesh(filename=mesh_path)
            self.vs.append(m.v)
            self.fs.append(m.f)
            tpl_edge_index = get_tpl_edges(m.v, m.f)
            self.tpl_edge_indexs.append(tpl_edge_index.astype(int).T)

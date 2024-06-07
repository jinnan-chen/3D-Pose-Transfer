import torch
import os 
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


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

gender='neutral'
path=os.getcwd()

model_path=os.path.abspath(os.path.join(os.getcwd(),'keypointdeformer','smpl_model', 'data','smpl'))
print(model_path)
if os.path.isdir(model_path):
    print('yes')
    model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
    smpl_path = os.path.join(model_path, model_fn)
with open(smpl_path, 'rb') as smpl_file:
    data_struct = Struct(**pickle.load(smpl_file,
                                    encoding='latin1'))

j_regressor = to_tensor(to_np(data_struct.J_regressor))
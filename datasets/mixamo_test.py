import itertools
import json
import os
from collections import Counter
from re import L
import traceback
from turtle import Vec2D
import warnings

import numpy as np
import pandas
import torch
from torch._six import container_abcs

from ..utils.utils import resample_mesh, normalize_to_box
from ..utils.io import read_keypoints, read_pcd, read_mesh, find_files


class mixamotest(torch.utils.data.Dataset):
    MESH_FILE_EXT = 'obj'
    POINT_CLOUD_FILE_EXT = 'pts'
    DO_NOT_BATCH = [
        'source_face', 'source_mesh', 'target_face', 'target_mesh', 'pair_face', 'pair_mesh', 
        'source_seg_points', 'target_seg_points', 'source_seg_labels', 'target_seg_labels', 'source_mesh_obj', 'target_mesh_obj','pair_mesh_obj']
    # CATEGORY2SYNSETOFFSET = {'airplane': '02691156', 'bag': '02773838', 'cap': '02954340', 'car': '02958343', 'chair': '03001627', 'earphone': '03261776', 'guitar': '03467517', 'knife': '03624134', 'lamp': '03636649', 'laptop': '03642806', 'motorbike': '03790512', 'mug': '03797390', 'pistol': '03948459', 'rocket': '04099429', 'skateboard': '04225987', 'table': '04379243'}
    # SYNSETOFFSET2CATEGORY = {v: k for k, v in CATEGORY2SYNSETOFFSET.items()}    

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--category', required=False,default='mixamo', type=str, help='')
        parser.add_argument('--segmentations_dir', type=str, default=None, help='')
        parser.add_argument('--seg_split_dir', type=str, default=None, help='')
        parser.add_argument('--keypointnet_dir', type=str, default=None, help='')
        parser.add_argument('--keypointnet_compatible', type=str, default=None, help='')
        parser.add_argument('--keypointnet_common_keypoints', action='store_true', help='')
        parser.add_argument("--keypointnet_min_n_common_keypoints", type=int, default=6, help="")
        parser.add_argument("--keypointnet_min_samples", type=float, default=0.8, help="")
        parser.add_argument('--keypoints_gt_source', type=str, default=None, help='')
        # parser.add_argument('--data_type', type=str, default='mixamo', help='')
        parser.add_argument('--split_file', type=str, default=None, help='')
        parser.add_argument('--split', type=str, default=None, help='')
        parser.add_argument("--fixed_source_index", type=int, default=None, help="")
        parser.add_argument("--fixed_target_index", type=int, default=None, help="")
        parser.add_argument('--normalize', type=str, default='unit_box', help='')
        parser.add_argument("--multiply", type=int, default=1, help="")
        parser.add_argument('--load_cages_test_pairs', action='store_true', help='')
        parser.add_argument('--load_test_pairs', action='store_true', help='')
        parser.add_argument('--load_mesh', default='False', help='')
        parser.add_argument('--sample_mesh', action='store_true', help='')
        parser.add_argument('--test_pairs_file', type=str, default=None, help='')
        parser.add_argument('--smpl_root', type=str, default='../mixamo6890testcen0520', help='')
        parser.add_argument('--smpl_list', type=str, default='../mixamo_testcen120.txt', help='')
        # parser.add_argument('--smpl_list2', type=str, default='/ssd/jnchen/smpl_unsup.txt', help='')
        parser.add_argument('--mixamo',action='store_true', help='')
        return parser

    def normalize(self, x):
        if self.opt.normalize == 'unit_box':
            pc, center, scale = normalize_to_box(x)
        else:
            raise ValueError()
        return pc, center, scale

    def __init__(self, opt):        
        self.opt = opt
        self.mesh_dir = opt.mesh_dir
        self.dataset= self.load_dataset()
        print("dataset size %d" % len(self))
    # def convert_ends(self,dic):
    #     dic_ends={}
    #     end2=0
    #     end1=0
    #     for key in dic:
    #         end2+=int(dic[key])
    #         # print(end1,end2)
    #         dic_ends[key]=[end1,end2]
    #         end1+=int(dic[key])
    #         # print(end2,end1)
    #     return dic_ends
    def _load_from_action_split(self, fileroot):
        # count_dic={}
        # charactors=sorted(os.listdir(mixamo_root))
        # count_dic=count_dic.fromkeys(charactors,0)
        with open(fileroot) as f:
            lines = f.read().splitlines()
        

        # for dirpath, dirnames , filenames in (os.walk(mixamo_root)):  
        #     for file in filenames:
        #         if os.path.splitext(file)[-1] == '.obj':  
        #             names.append(os.path.join(dirpath,file))
        #             cha_name=dirpath.split('/')[-2]
        #             count_dic[cha_name]=count_dic[cha_name]+1
        # count_dic=self.convert_ends(count_dic)
        # for charactor in charactors:
        #     for action in os.listdir(os.path.join(mixamo_root,charactor)):
        #         for frame in os.path.joi
        # find names from the category and split
        return lines
    def _load_from_split_file(self, split):
        df = pandas.read_csv(self.opt.split_file)
        # find names from the category and split
        df = df.loc[(df.synsetId == int(self.opt.category)) & (df.split == self.opt.split)]
        names = df.modelId.values
        return names

    def _load_from_files(self):
        files = find_files(os.path.join(self.opt.points_dir, self.opt.category), self.POINT_CLOUD_FILE_EXT)
        # extract name from files
        names = [x.split(os.path.sep)[-2] for x in files]
        names = sorted(names)
        return names

    def _load_seg_split_file(self, seg_split_file):
        with open(seg_split_file) as f:
            files = json.load(f)
            # ['04379243', '9db8f8c94dbbfe751d742b64ea8bc701'], ['02691156', '329a018e131ece70f23c3116d040903f'], ...
            names = [x.split(os.path.sep)[-2:] for x in files]
            # filter out other categories
            names = [x[1] for x in names if x[0] == self.opt.category] 
        return names

    def _load_seg_split(self):
        seg_split_file = os.path.join(self.opt.seg_split_dir, 'shuffled_%s_file_list.json' % self.opt.split)
        return self._load_seg_split_file(seg_split_file)


    def _load_keypointnet_split(self, split_name):
        # load split
        with open(os.path.join(self.opt.keypointnet_dir, 'splits', split_name + '.txt')) as f:
            lines = f.read().splitlines()
        # line looks like this: 02691156-ecbb6df185a7b260760d31bf9510e4b7
        split = set([x[len(self.opt.category) + 1:] for x in lines if x.startswith(self.opt.category)])
        return split

    def _load_keypointnet(self):
        # load keypoints
        file_path = os.path.join(self.opt.keypointnet_dir, 'annotations', self.SYNSETOFFSET2CATEGORY[self.opt.category] + '.json')
        with open(file_path) as f:
            data = json.load(f)
        keypoints = {}
        for item in data:
            name = item['model_id']
            keypoints_sample = [x['xyz'] for x in item['keypoints']]
            keypoint_ids_sample = [x['semantic_id'] for x in item['keypoints']]
            keypoints_sample = np.array(keypoints_sample, dtype=np.float32)
            keypoints[name] = (keypoints_sample, keypoint_ids_sample)
        if self.opt.keypointnet_common_keypoints:
            # get most common keypoint ids
            ids = [list(id) for _, id in keypoints.values()]
            max_keypoints = len(set(itertools.chain(*ids)))
            # start with the highest number of keypoints
            success = False
            for n_common_keypoints in range(max_keypoints, self.opt.keypointnet_min_n_common_keypoints, -1):
                most_common_ids = sorted([x[0] for x in Counter(itertools.chain(*ids)).most_common(n_common_keypoints)])
                # prune keypoints
                pruned_keypoints = {}
                for name, (sample_keypoints , id) in keypoints.items():
                    if set(most_common_ids).issubset(id):
                        indices = [ id.index(x) for x in most_common_ids]
                        new_keypoints = sample_keypoints[indices]
                        pruned_keypoints[name] = new_keypoints
                if len(pruned_keypoints) / len(keypoints) > self.opt.keypointnet_min_samples:
                    success=True
                    break
            if not success:
                raise ValueError()
            keypoints = pruned_keypoints
        else:
            keypoints = {k: v[0] for k, v in keypoints.items()}

        return keypoints


    def _get_shapenet_id_to_model_id(self):
        df = pandas.read_csv(self.opt.split_file)
        return {k: v for k, v in zip(df.id, df.modelId)}

    
    def _load_test_pairs(self):
        with open(self.opt.test_pairs_file, 'r') as f:
            lines = f.read().splitlines()
        names = []
        partners = []
        for line in lines:
            name = line.split(' ')[0]
            partner = line.split(' ')[1]
            names += [name]
            partners += [partner]
        return names, partners


    def load_dataset(self):
        dataset = {}
        
        names = self._load_from_action_split(self.opt.smpl_list)
        # if self.opt.data_type == 'shapenet':
        #     names = self._load_from_split_file(self.opt.split)
        # elif self.opt.data_type == 'keypointnet':
        #     keypoints = self._load_keypointnet()
        #     names = list(self._load_keypointnet_split(self.opt.split))
        #     dataset['keypoints'] = keypoints
        # elif self.opt.data_type == 'shapenetseg':
        #     names = self._load_seg_split()
        # elif self.opt.data_type == 'files':
        #     names = self._load_from_files()
        # else:
        #     raise ValueError()

        # if self.opt.keypoints_gt_source == 'keypointnet':
        #     keypoints = self._load_keypointnet()
        #     names = [x for x in names if x in keypoints]
        #     dataset['keypoints'] = keypoints
        
        assert len(names) > 0
        
        # if self.opt.keypointnet_compatible:
        #     if self.opt.split == 'train':
        #         # remove keypointnet val and test from training
        #         val_split  = self._load_keypointnet_split('val')
        #         test_split = self._load_keypointnet_split('test')
        #         names = set(names)
        #         names -= val_split
        #         names -= test_split
        
        names = sorted(list(names))
        
        
        if self.opt.load_test_pairs:
            names, partners = self._load_test_pairs()
            dataset['partners'] = partners
        else:
            names = sorted(list(names))

        dataset['name'] = names

        return dataset


    def _get_pointcloud_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "model.pts")

    def _get_mesh_path(self, name):
        return os.path.join(self.opt.mesh_dir, self.opt.category, name, "model.obj")
    def _get_mixamo_path(self, name):
        return os.path.join(name)

    def _get_keypoints_path(self, name):
        return os.path.join(self.opt.keypoints_dir, self.opt.category, name, "keypoints.txt")


    def _get_seg_points_path(self, name):
        return os.path.join(self.opt.segmentations_dir, self.opt.category, 'points', name + '.pts')


    def _get_seg_labels_path(self, name):
        return os.path.join(self.opt.segmentations_dir, self.opt.category, 'points_label', name + '.seg')

    
    def _read_keypointnet_keypoints(self, name):
        keypoints = torch.from_numpy(self.dataset['keypoints'][name]).float()
        
        # fix axis
        keypoints = keypoints[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]]) 

        # compensate for their normalization
        # load associated point cloud
        pcd_path = os.path.join(self.opt.keypointnet_dir, 'pcds', self.opt.category, name + '.pcd')
        points = read_pcd(pcd_path)
        points = torch.from_numpy(points).float()
        points = points[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]]) 
        _, center, scale = self.normalize(points)
        keypoints = (keypoints - center) / scale

        return keypoints, center[0], scale[0]
    

    def _read_txt_keypoints(self, name):
        keypoints = read_keypoints(self._get_keypoints_path(name))
        keypoints = torch.from_numpy(keypoints).float()
        return keypoints


    def get_item(self, index):
        return self.get_item_by_name(self.dataset['name'][index])


    def get_item_by_name(self, name, sample_mesh=False, load_mesh=False,mixamo=True):        
        if mixamo:
            # print(name)
            
            mesh_path = self._get_mixamo_path(name)
            try:
                V_mesh, F_mesh, mesh_obj = read_mesh(mesh_path, return_mesh=True)
            except Exception:  
                print(name)          
        # V_mesh[:1]-=0.2
        # if load_mesh:
        #     mesh_path = self._get_mesh_path(name)
        #     V_mesh, F_mesh, mesh_obj = read_mesh(mesh_path, return_mesh=True)

        # if sample_mesh:
        #     points = resample_mesh(mesh_obj, self.opt.num_point)
            # print(points.shape)
        # else:
        points=V_mesh
            # load points sampled from a mesh
            # points = np.loadtxt(self._get_pointcloud_path(name), dtype=np.float32)
            # points = torch.from_numpy(points).float()
        # points[:, :3], center, scale = self.normalize(points[:, :3])
        points = points.clone()
        normals = points[:,3:6].clone()
        label = points[:,-1].clone()
        shape = points[:,:3].clone()
        
        scale=1
        cha_name=name.split('/')[-1].split('_')[0]
        # print(cha_name)
        if cha_name=='Dreyar':
            scale*=10
            
        elif cha_name in ['Liam' ,'Malcolm' , 'Remy','Shae', 'Suzie','stefani']:
            scale*=2
        if mixamo:
            V_mesh = V_mesh[:,:3]
            F_mesh = F_mesh[:,:3]
            V_mesh = (V_mesh ) / scale
            result = {'shape': V_mesh, 'normals': normals, 'label': label, 'cat': self.opt.category, 'file': name}
            result.update({'mesh': V_mesh, 'face': F_mesh, 'mesh_obj': mesh_obj})

        # load labels    
        if self.opt.keypoints_gt_source == 'keypointnet':
            keypoints_gt, keypoints_gt_center, keypoints_gt_scale = self._read_keypointnet_keypoints(name)
            result['keypoints_gt'] = keypoints_gt
            result['keypoints_gt_center'] = keypoints_gt_center
            result['keypoints_gt_scale'] = keypoints_gt_scale

        # if self.opt.data_type == 'shapenetseg':
        #     assert self.opt.segmentations_dir is not None
        #     seg_points = np.loadtxt(self._get_seg_points_path(name)).astype(np.float32)
        #     seg_labels = np.loadtxt(self._get_seg_labels_path(name)).astype(np.int32)
        #     seg_points = torch.from_numpy(seg_points)
        #     seg_labels = torch.from_numpy(seg_labels)
        #     seg_points = (seg_points - center) / scale
        #     result.update({'seg_labels': seg_labels, 'seg_points': seg_points})

        return result
    

    def get_sample(self, index):

        names = self.dataset['name'][index]

        
        # id_mesh_name = data_list[0].replace('/','_')+  "_%06d"%(num+1)+'.obj'
        # pose_mesh_name = data_list[1].strip(' ').replace('/','_')+ "_%06d"%(num+1)+'.obj'
        # gt_mesh_name = data_list[2].strip(' ').replace('/','_')+ "_%06d"%(num+1)+'.obj'
        num=0
        # import ipdb
        # ipdb.set_trace()
        name,name_2,name_3,_=names.strip('\n').split('.fbx')
        name=os.path.join(self.opt.smpl_root,name.replace('/','_')+  "_%06d"%(num+1)+'.obj')
        name_2=os.path.join(self.opt.smpl_root,name_2.strip(' ').replace('/','_')+ "_%06d"%(num+1)+'.obj')
        name_3=os.path.join(self.opt.smpl_root,name_3.strip(' ').replace('/','_')+ "_%06d"%(num+1)+'.obj')
        # print(name,name_2,name_3)
        # ind1,ind2=self.counts_dic[cha_name]
        # max_n=self.get_real_length()
        # p1=ind1/(max_n+ind1-ind2)
        # p2=1-p1
        # index_3 = np.random.randint(ind1,ind2)
        # list2=list(self.counts_dic.keys() )
        # list2.remove(cha_name)
        
        # name_2=np.random.choice(list2)
        # ind3,ind4=self.counts_dic[name_2]
        # index_2=np.random.randint(ind3,ind4)

        # name_2 = self.dataset['name'][index_2]
        # name_3 = self.dataset['name'][index_3]
        # import ipdb
        # ipdb.set_trace()
        sample_mesh =False
        source_data = self.get_item_by_name(name, load_mesh=self.opt.mixamo, sample_mesh=sample_mesh)
        target_data = self.get_item_by_name(name_2, load_mesh=self.opt.mixamo, sample_mesh=sample_mesh)

        source_pair_data = self.get_item_by_name(name_3, load_mesh=self.opt.mixamo, sample_mesh=sample_mesh)
        
        result = {'source_' + k: v for k, v in source_data.items()}
        result.update({'target_' + k: v for k, v in target_data.items()})
        result.update({'pair_' + k: v for k, v in source_pair_data.items()})
        # print(result)
        return result


    def __getitem__(self, index):
        # return self.get_sample(index)

        for _ in range(10):
            index = index % self.get_real_length()
            # index = 1000000
            try:
                # print(self.get_s
                # ample(index))
                return self.get_sample(index)
            # except Exception as e:
            except:
                print('ee')
                # warnings.warn(f"Error loading sample {index}: " + ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                # import ipdb; ipdb.set_trace()
                index += 1


    @classmethod
    def collate(cls, batch):
        
        
        batched = {}
        # print(batch)
        elem = batch[0]
        for key in elem:
            if key in cls.DO_NOT_BATCH:
                batched[key] = [e[key] for e in batch]
            else:
                try:
                    batched[key] = torch.utils.data.dataloader.default_collate([e[key] for e in batch])
                except Exception as e:
                    print(e)
                    print(key)
                    import ipdb; ipdb.set_trace()
                    print()

        return batched


    @staticmethod
    def uncollate(batched):
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.cuda(device=0)
            elif isinstance(v, container_abcs.Sequence):
                if isinstance(v[0], torch.Tensor):
                    batched[k] = [e.cuda() for e in v]
        return batched


    def get_real_length(self):
        return len(self.dataset['name'])


    def __len__(self):
        if self.opt.fixed_target_index is not None:
            return 1000
        else:
            return self.get_real_length() * self.opt.multiply

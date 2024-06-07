import json
import os
import time
from datetime import datetime
from sklearn.utils import compute_class_weight
import torch.nn as nn
import numpy as np
import pytorch3d.io
import torch
import torch.nn.parallel
import torch.utils.data

from keypointdeformer.datasets import get_dataset
from keypointdeformer.models import get_model
from keypointdeformer.options.base_options import BaseOptions
from keypointdeformer.utils import io
from keypointdeformer.utils.gmm import deform_with_GMM
from keypointdeformer.utils.nn import load_network, save_network, weights_init
from keypointdeformer.utils.utils import Timer
from tensorboardX import SummaryWriter

CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINT_EXT = '.pth'


def write_losses(writer, losses, step):
    for name, value in losses.items():
        writer.add_scalar('loss/' + name, value, global_step=step)


def save_normalization(file_path, center, scale):
    with open(file_path, 'w') as f:
        json.dump({'center': [str(x) for x in center.cpu().numpy()], 'scale': str(scale.cpu().numpy()[0])}, f)


def save_data_keypoints(data, save_dir, name):
    if name in data:
        io.save_keypoints(os.path.join(save_dir, name + '.txt'), data[name])


def save_data_txt(f, data, fmt):
    np.savetxt(f, data.cpu().detach().numpy(), fmt=fmt)


def save_pts(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    io.save_pts(f, points.cpu().detach().numpy(), normals=normals)


# def save_ply(f, verts, faces):
#     pytorch3d.io.save_ply(f, verts.cpu(), faces=faces.cpu())

    
def save_output(cycle_times,save_dir_root, data, outputs, save_mesh=True, save_auxilary=True):
    # print(save_dir_root)
    name = data['source_file']
    name2 = data['target_file']
    # print('name',name.split('/')[-3]+name.split('/')[-1][:-4])
    # import ipdb

    # ipdb.set_trace()
    name_short=name.split('/')[-1][:-4]+'_'+name2.split('/')[-1][:-4]
    save_dir = os.path.join(save_dir_root, name_short+'_'+cycle_times)
    # print(save_dir)
    os.makedirs(save_dir,exist_ok=True)

    # save meshes
    # save_mesh=True
    # if save_mesh and 'source_mesh' in data:
    #     io.save_mesh(os.path.join(save_dir, 'source_mesh.obj'), data["source_mesh"], data["source_face"])
    #     # print('savemesh')
    #     if save_auxilary:
    #         save_data_txt(os.path.join(save_dir, 'source_vertices.txt'), data["source_mesh"], '%0.6f')
    #         save_data_txt(os.path.join(save_dir, 'source_faces.txt'), data["source_face"], '%d')

    #     io.save_mesh(os.path.join(save_dir, 'target_mesh.obj'), data["target_mesh"], data["target_face"])
    # print(cycle_times)
    if cycle_times=='test':
        io.save_mesh(os.path.join(save_dir, 'gt_mesh.obj'), data['pair_shape'], data["pair_face"])
    if outputs is not None:
        io.save_mesh(os.path.join(save_dir, 'source_mesh.obj'), outputs['source'], outputs["source_face"])
        io.save_mesh(os.path.join(save_dir, 'target_mesh.obj'), outputs['target'], outputs["target_face"])
        io.save_mesh(os.path.join(save_dir, 'deformed_mesh.obj'), outputs['deformed'], outputs["source_face"])
        io.save_mesh(os.path.join(save_dir, 'deformed_mesh_c.obj'), outputs['deformed_c'], outputs["source_face"])
        # io.save_mesh(os.path.join(save_dir, 'pair_mesh.obj'), outputs['pair'], outputs["pair_face"])
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'weights.txt'), outputs['weights'], '%0.6f')
            # print(outputs['weights'][1].max())

    # save pointclouds
    # save_pts(os.path.join(save_dir, 'source_pointcloud.pts'), data['source_shape'], normals=data['source_normals'])
    # if outputs is not None:
    #     save_pts(os.path.join(save_dir, 'deformed_pointcloud.pts'), outputs['deformed'])
    #     # if save_auxilary:
    #     #     save_data_txt(os.path.join(save_dir, 'influence.txt'), outputs['influence'], '%0.6f')

    # save_pts(os.path.join(save_dir, 'target_pointcloud.pts'), data['target_shape'], normals=data['target_normals'])
    
    # save cages
    # if outputs is not None:
        # save_ply(os.path.join(save_dir, 'cage.ply'), outputs["cage"], outputs["cage_face"])
        # if save_auxilary:
        #     save_data_txt(os.path.join(save_dir, 'cage.txt'), outputs["cage"], '%0.6f')
        # save_ply(os.path.join(save_dir, 'deformed_cage.ply'), outputs["new_cage"], outputs["cage_face"])
    
    if outputs is not None:
        io.save_keypoints(os.path.join(save_dir, 'source_keypoints.txt'), outputs["source_keypoints"].transpose(0, 1))
        io.save_keypoints(os.path.join(save_dir, 'source_keypoints_deformed.txt'), outputs["source_keypoints_deformed"].transpose(0, 1))
        io.save_keypoints(os.path.join(save_dir, 'target_keypoints.txt'), outputs["target_keypoints"].transpose(0, 1))
    
    save_data_keypoints(data, save_dir, 'source_keypoints_gt')
    save_data_keypoints(data, save_dir, 'target_keypoints_gt')

    # io.save_keypoints(os.path.join(save_dir, 'source_init_keypoints.txt'), outputs['source_init_keypoints'].transpose(0, 1))
    # io.save_keypoints(os.path.join(save_dir, 'target_init_keypoints.txt'), outputs['target_init_keypoints'].transpose(0, 1))
        
        # if 'source_keypoints_gt_center' in data:
        #     save_normalization(os.path.join(save_dir, 'source_keypoints_gt_normalization.txt'), data['source_keypoints_gt_center'], data['source_keypoints_gt_scale'])

        # if 'source_seg_points' in data:
        #     io.save_labelled_pointcloud(os.path.join(save_dir, 'source_seg_points.xyzrgb'), data['source_seg_points'].detach().cpu().numpy(), data['source_seg_labels'].detach().cpu().numpy())
        #     save_data_txt(os.path.join(save_dir, 'source_seg_labels.txt'), data['source_seg_labels'], '%d')


def split_batch(data, b, singleton_keys=[]):
    return {k: v[b] if k not in singleton_keys else v[0] for k, v in data.items()}


def save_outputs(cycle_times,outputs_save_dir, data, outputs, save_mesh=True):
    for b in range( data['source_shape'].shape[0] ):
        save_output(cycle_times,
            outputs_save_dir, split_batch(data, b, singleton_keys=['cage_face']), 
            split_batch(outputs, b, singleton_keys=['cage_face']), save_mesh=save_mesh)
def get_data_test(dataset, data):
    data = dataset.uncollate(data)
    source_shape, target_shape = data["source_shape"], data["target_shape"]
    # tsource_shape, ttarget_shape,tpair_shape = data["source_tshape"].transpose(1, 2), data["target_tshape"].transpose(1, 2),data["pair_tshape"].transpose(1, 2)
    source_face=data["source_face"]
    target_face=data["target_face"]
    # source_mesh=data["source_mesh_obj"]
    # target_mesh=data["target_mesh_obj"]

    # pair_mesh=data["pair_mesh_obj"]
    pair_face=data["pair_face"]
    pair_shape_t=data["pair_shape"]
    
    source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)
    pair_shape_t=pair_shape_t.transpose(1, 2)
    
    return   source_face,target_face,pair_face, source_shape_t, target_shape_t,pair_shape_t


def get_data(dataset, data):
    data = dataset.uncollate(data)
    source_shape, target_shape = data["source_shape"], data["target_shape"]
    tsource_shape, ttarget_shape,tpair_shape = data["source_tshape"].transpose(1, 2), data["target_tshape"].transpose(1, 2),data["pair_tshape"].transpose(1, 2)
    source_face=data["source_face"]
    target_face=data["target_face"]
    # source_mesh=data["source_mesh_obj"]
    # target_mesh=data["target_mesh_obj"]

    # pair_mesh=data["pair_mesh_obj"]
    pair_face=data["pair_face"]
    pair_shape_t=data["pair_shape"]
    
    source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)
    pair_shape_t=pair_shape_t.transpose(1, 2)
    
    return   source_face,target_face,pair_face, source_shape_t, target_shape_t,pair_shape_t,tsource_shape, ttarget_shape,tpair_shape 


def test(opt, save_subdir="test"):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
        collate_fn=dataset.collate,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    net = get_model(opt.model)(opt).cuda()
    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
    load_network(net, ckpt)
    
    net.eval()

    test_output_dir = os.path.join(log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    timer = Timer('step')
    PMD=0.0
    Chamfer=0.0
    PMD_C=0.0
    PMD_=[]
    num=0
    mse=torch.nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            timer.stop()
            timer.start()
            # data
            num+=1
            data = dataset.uncollate(data)
            source_face,target_face,pair_face, source_shape_t, target_shape_t,pair_shape_t= get_data_test(dataset, data)
            t1=time.time()
            outputs = net (source_shape_t,source_face,target_face,source_shape_t, target_shape_t)
            # print('',time.time()-t1)
            pair_shape_t=pair_shape_t.float()
            cham=pytorch3d.loss.chamfer_distance(outputs['deformed'], pair_shape_t.permute(0,2,1))[0]
            Chamfer+=cham
            pmd=mse(pair_shape_t.permute(0,2,1),outputs['deformed'])
            pmd_c=mse(pair_shape_t.permute(0,2,1),outputs['deformed_c'])
            PMD+=pmd
            PMD_C+=pmd_c
            PMD_.append(pmd)
            # if pmd ==min(PMD_):
            print(data['source_file'],data['target_file'],pmd)
            print(PMD/num*100)
            # print(cham)  
            save_outputs('test',os.path.join(log_dir, save_subdir), data, outputs)
    print('LOSS_PMD:',PMD/dataset.get_real_length()*100)
    print('LOSS_PMD_C:',PMD_C/dataset.get_real_length()*100)
    print('LOSS_Cham:',Chamfer/dataset.get_real_length()*100) 
from tqdm import tqdm, trange
def train(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate,
        num_workers=opt.n_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    
    net = get_model(opt.model)(opt)
    
    net.apply(weights_init)

    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    # train
    net.train()
    t = 0

    # train
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")
    # import ipdb
    # ipdb.set_trace()
    log_file.write(str(opt) + "\n")

    summary_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir=os.path.join(checkpoints_dir, 'logs', summary_dir), flush_secs=5)

    if opt.iteration:
        t = opt.iteration
    
    iter_time_start = time.time()
    
    for ep in tqdm( torch.arange(0,150)):
        while t <= opt.n_iterations:
            for _, data in enumerate(tqdm(dataloader)):
                if t > opt.n_iterations:
                    break
                source_face,target_face,pair_face, source_shape_t, target_shape_t,pair_shape_t,tsource_shape, ttarget_shape,tpair_shape = get_data(dataset, data)
                # print(source_v[1].shape,target_v[1].shape,pair_v[1].shape)
                # outputs = net (tsource_shape,source_face,target_face,source_shape_t, target_shape_t)
                # current_loss = net.compute_loss(t)
                
            # B*6890*3
            # outputs['deformed']=outputs['deformed'].transpose(2,1)
            

            
                # if t > opt.iterations_init_points:

                if opt.lambda_sup>0:    
                    outputs_sup=net(ttarget_shape ,target_face, source_face ,target_shape_t,source_shape_t  )
                    current_loss = net.compute_loss(t)
            
                    net.compute_loss_sup( current_loss,pair_shape_t,outputs_sup)
                else:
                    outputs = net (tsource_shape,source_face,target_face,source_shape_t, target_shape_t)
                    current_loss = net.compute_loss(t)
                    outputs2=net(tpair_shape, pair_face, source_face ,pair_shape_t,outputs['deformed'].transpose(1, 2)  )
                    net.compute_loss_cycle(current_loss,target_shape_t.float(),target_face,outputs2,outputs['source_keypoints_deformed'],t  )
                    
                    outputs3=net( ttarget_shape,target_face, pair_face ,target_shape_t,pair_shape_t  )
                    net.compute_loss_pair(current_loss,pair_shape_t.float(),pair_face,outputs3 ,t)
                # import ipdb
                # ipdb.set_trace()
                # for k in current_loss:
                #     current_loss[k]=current_loss[k].float()
                net.optimize(current_loss, t)
                if opt.lambda_sup==0:    
                    if t % opt.save_interval == 0:

                        outputs_save_dir = os.path.join(checkpoints_dir, 'outputs', '%07d' % t)
                        if not os.path.exists(outputs_save_dir):
                            os.makedirs(outputs_save_dir)
                        save_outputs('1',outputs_save_dir, data, outputs, save_mesh=False)
                        if t > opt.iterations_init_points:
                            save_outputs('2',outputs_save_dir, data, outputs2, save_mesh=False)
                        save_network(net, checkpoints_dir, network_label="net", epoch_label=t)

                iter_time = time.time() - iter_time_start
                iter_time_start = time.time()
                if (t % opt.log_interval == 0):
                    lr=net.scheduler.get_last_lr()
                    log_str = ''
                    samples_sec = opt.batch_size / iter_time
                    losses_str = ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])
                    log_str = "lr:"+str(lr)+" {:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                        t, iter_time, samples_sec, losses_str)

                    tqdm.write(log_str)
                    log_file.write(log_str + "\n")

                    write_losses(writer, current_loss, t)
                t += 1
            # if t%2000==0:
            # ep+=1

    log_file.close()
    save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = BaseOptions()
    
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        test(opt, save_subdir=opt.subdir)
    elif opt.phase == "train":
        print(opt)
        train(opt)
    else:
        raise ValueError()

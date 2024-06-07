import bpy
import numpy as np
from os import listdir, path
import os
def fbx2bvh(data_path,desti_path, file):
    sourcepath = data_path+"/"+file
    if not os.path.exists(desti_path):
      os.mkdir(desti_path)
    if not os.path.exists(desti_path+"/"+file.split(".fbx")[0]):
      os.mkdir(desti_path+"/"+file.split(".fbx")[0])
    obj_path = desti_path+"/"+file.split(".fbx")[0]+"/"+file.split(".fbx")[0]+".obj"
    
    # if i==0:
    while bpy.data.objects:
      bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=sourcepath)
    # objs = [bpy.context.scene.objects['Camera'], bpy.context.scene.objects['Cube']]
    # bpy.ops.object.delete({"selected_objects": objs})
    # frame_start = 9999
    # frame_end = -9999
    action = bpy.data.actions[-1]
    # if  action.frame_range[1] > frame_end:
    #   frame_end = action.frame_range[1]
    # if action.frame_range[0] < frame_start:
    #   frame_start = action.frame_range[0]

    # frame_end = np.max([60, frame_end])
    bpy.data.scenes[0].frame_end=action.frame_range[1]
    bpy.data.objects[0].select_set(True)
    print('sss',bpy.data.objects[0])
    bpy.ops.export_scene.obj(filepath=obj_path,
                            use_materials=False,
                            use_animation=True)
                            # root_transform_only=True)
    # bpy.data.actions.remove(bpy.data.actions[-1])
    # print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    data_path = "/home/jin/Downloads/mixamo_dataset"
    desti_path="/home/jin/Downloads/mixamo_dataset_obj"
    directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
    
    for d in directories[17:]:

      files_ori = sorted([f for f in listdir(path.join(data_path,d)) if f.endswith(".fbx")])
      files=files_ori[:30]
      for file in files:
          fbx2bvh(path.join(data_path,d), path.join(desti_path,d), file)
        
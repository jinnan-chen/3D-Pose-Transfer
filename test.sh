# CUDA_VISIBLE_DEVICES=2 python scripts/main.py --phase test --name te_faust_tpose_0_0816 --n_keypoints 24  --n_workers 0 --dataset dfaust --batch_size 1 --gl 'one'  --n_input 27  --tw --refine  --ckpt  '/ssd/jnchen/kd_log/faust_0815_23matrix_input27_edge0.0005_kp0.1_decay0.3_glone_tw_tp0/checkpoints/net_final.pth'
# CUDA_VISIBLE_DEVICES=2 python scripts/main.py --phase test --tw --name te_mixamo_trsmpl_bounding --n_keypoints 24 --n_workers 0 --dataset mixamo_test --refine --batch_size 1 --gl 'one' --n_input 27  --ckpt '/ssd/jnchen/kd_log/smpl_0808_23matrix_input27_edge0.0005_kp0.4_decay0.3_glone_twsin0_tw_bounding/checkpoints/net_final.pth'
# CUDA_VISIBLE_DEVICES=0 python scripts/main.py --phase test  --name smpl_test_0130_decay0.5pose_aug400 --dataset smpl_test_3dpt --batch_size 1 --n_keypoints 24 --gl 'one'  --n_input 27  --tw --refine  --ckpt '/ssd/jnchen/kd_log/smpl_0128_23matrix_input27_edge0.0005_kp0.4_decay_0.3_glone_twsin0_tw_lr_augu/checkpoints/net_29000.pth'
# CUDA_VISIBLE_DEVICES=1 python scripts/main.py --phase test  --name mixamo_test_0817 --dataset mixamo_test  --batch_size 1  --n_keypoints 24  --gl 'one'  --n_input 27  --tw --refine  --ckpt '/ssd/jnchen/deformer_log/mixamo0808_23_27_three0.0005kp0.1_tpose_tw_glone/checkpoints/net_final.pth'
# CUDA_VISIBLE_DEVICES=1 python scripts/main.py --phase test  --name smal_test_0201_kp0.04 --dataset smal_test --batch_size 1 --n_keypoints 33 --gl 'one'  --n_input 36  --tw --refine  --smal --ckpt '/ssd/jnchen/kd_log/smal_0130_32matrix_input36_edge0.0005_kp0.04_decay_0.5_glone_twsin0_tw_lr5e-4/checkpoints/net_58000.pth'
CUDA_VISIBLE_DEVICES=0 python scripts/main.py --phase test  --name smpl_test_0528_twkpt --dataset smpl_test --batch_size 1 --n_keypoints 24 --gl 'one'  --n_input 27  --tw --refine  --ckpt '/ssd/jnchen/kd_log/smpl_0528_23matrix_input27_edge0.0005_kp0.4_decay_0.5_glone_twsin0_tw_twkp/checkpoints/net_final.pth'
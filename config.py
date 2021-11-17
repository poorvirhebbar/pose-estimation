import os
import os.path as osp
import sys
import numpy as np

class Config:

    ## dataset
    trainset = ['Human36M'] #, 'MPII']
    testset = 'Human36M' # Human36M, MPII

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir =  '/home/datasets/rahulm_projects/Human3.6/images'     # osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    ## model setting
    resnet_type = 50 # 18, 34, 50, 101, 152

    ## input, output
    input_shape = (256, 256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    lr_dec_epoch = [15, 17]
    end_epoch = 20
    hidden_dim = 512    # hidden dim of GPT decoder
    lr = 1e-5
    lr_dec_factor = 0.1
    optimizer = 'adam'
    weight_decay = 1e-5
    batch_size = 32

    ## testing config
    test_batch_size = 32
    flip = 0

    ## others
    num_thread = 20 #8
    gpu_ids = '33'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()
resnet_type = 50
'''
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
'''

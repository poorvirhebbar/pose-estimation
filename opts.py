import argparse
import os


class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

  def init(self):
    self.parser.add_argument('--exp_id', default='default', help='Experiment ID')
    self.parser.add_argument('--test', action='store_true', help='test')
    #self.parser.add_argument('--data_dir', default='data/h36m/annotations', help='data directory')
    #self.parser.add_argument('--img_dir', default='data/h36m/images', help='image directory')
    #self.parser.add_argument('--pascal_voc_dir', default='/home/rahul/project/human-pose/synthetic-occlusion/VOCdevkit/VOC2012', help='image directory')

    self.parser.add_argument('--load_model', default=None, help='Provide full path to a previously trained model')
    self.parser.add_argument('--lr', type=float, default=1.0e-4, help='Learning Rate')
    self.parser.add_argument('--wd', type=float, default=1.0e-3, help='Weight Decay')
    self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    self.parser.add_argument('--betas', type=tuple, default=(0.5,0.9), help='betas for Adam')
    self.parser.add_argument('--break_points', type=tuple, default=(5,12,20), help='break points for LR update')
    self.parser.add_argument('--dropLR', type=int, default=20, help='drop LR')
    self.parser.add_argument('--flip', type=int, default=0, help='Whether to flip or not')
    self.parser.add_argument('--n_epochs', type=int, default=500, help='#training epochs')
    self.parser.add_argument('--start_epoch', type=int, default=0, help='First epoch number')
    self.parser.add_argument('--DEBUG', type=int, default=0, help='debug flag')
    self.parser.add_argument('--display', type=int, default=0, help='display message at iter')
    self.parser.add_argument('--val_intervals', type=int, default=1, help='#valid intervel')
    self.parser.add_argument('--save_intervals', type=int, default=100, help='#valid intervel')
    self.parser.add_argument('--n_iters', type=int, default=1000, help='#valid intervel')
    self.parser.add_argument('--train_batch', type=int, default=2, help='Mini-batch size')
    self.parser.add_argument('--val_batch', type=int, default=128, help='Mini-batch size')
    self.parser.add_argument('--chunk_size', type=int, default=12, help='Mini-batch size')
    self.parser.add_argument('--seed', type=int, default=0, help='RNG seed')

    self.parser.add_argument('--seq_len', type=int, default=500, help='Sequence length in wav2vec2 model')
    self.parser.add_argument('--seq_len_out', type=int, default=10, help='Prediction sequence length ')

    self.parser.add_argument('--reg_wt', type=float, default=0.1, help='weight for regulariser')
    self.parser.add_argument('--emb_wt', type=float, default=1.0, help='weight for metric learning')

    self.parser.add_argument('--arch', default='resnet18', help='resnet18 | resnet34 | ...')

    self.parser.add_argument('--optimizer', default='Adam', help='Adam | SGD | RMSprop | ...')
    self.parser.add_argument('--n_joints', type=int, default=25, help='num joints')

    self.parser.add_argument('--hidden_dim', type=int, default=2048, help='hidden dimension size in decoder')
    self.parser.add_argument('--n_decoders', type=int, default=4, help='number of decoder in GPT')
    self.parser.add_argument('--n_encoders', type=int, default=4, help='number of encoders in BERT')
    self.parser.add_argument('--n_codebook', type=int, default=1, help='Number of codebook(s) in wav2vec2')
    self.parser.add_argument('--n_quantized', type=int, default=1024, help='number of entries in codebook')
    self.parser.add_argument('--quant_dim', type=int, default=1024, help='quantized vector dim')
    self.parser.add_argument('--output_embed_dim', type=int, default=2048, help='output embed dim')
    self.parser.add_argument('--embed_dim', type=int, default=256, help='pose embedding dim')
    self.parser.add_argument('--encoder_attention_heads', type=int, default=3, help='Number of attention =heads')

    self.parser.add_argument('--layer_norm_first', type=bool, default=False)
    self.parser.add_argument('--conv_pos', type=int, default=31)

    self.parser.add_argument('--dataset', default='h36m', help='h36m | mpi_inf')

    self.parser.add_argument('--no_emb', action='store_true', help='disables embedding loss')
    self.parser.add_argument('--no_pose', action='store_true', help='disables pose loss')
    self.parser.add_argument('--no_image', action='store_true', help='disables image loading')
    self.parser.add_argument('--no_mean_bone', action='store_true', help='disables mean bone len loss')
    self.parser.add_argument('--get_embed', action='store_true', help='save embeddings from BERT')

    self.parser.add_argument('--sch_emb', type=int, default=1, help='schedule for emb loss')
    self.parser.add_argument('--sch_pose', type=int, default=1, help='schedule for emb loss')

    self.parser.add_argument('--mask_prob', default=0.3, type=float, help='masking prob of time step in wav2vec2')
    self.parser.add_argument('--dropout', default=0.2, type=float, help='Dropout in decoder')

  def parse(self):
    self.init()
    self.opt = self.parser.parse_args()
    self.opt.save_dir = os.path.join('./exp', self.opt.exp_id)
    os.system(f'mkdir -p {self.opt.save_dir}')
    #self.opt.data_dir = self.opt.data_dir
    # self.opt.downSample = list(map(int,self.opt.downSample.strip('[]').split(',')))
    #self.opt.sub_list_reg = list(map(int, self.opt.sub_list_reg.strip('[]').split(',')))
    #self.opt.sub_list_met = list(map(int, self.opt.sub_list_met.strip('[]').split(',')))
    # self.opt.camPairs = [list(map(int,x.strip(',(').split(',')))
    # for x in self.opt.camPairs.strip(')]').strip('[').split(')')]

    if self.opt.test is True:
      assert self.opt.load_model is not None

    return self.opt


def save_opt_make_dirs(opt):
    args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))

    if opt.test is False:
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        if not os.path.exists(os.path.join(opt.save_dir, 'visualize')):
            os.makedirs(os.path.join(opt.save_dir, 'visualize'))

        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')
        # for k, v in sorted(refs.items()):
        #   opt_file.write('  %s: %s\n' % (str(k), str(v)))

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import copy
LOG = True

#import numpy as np
#import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
#        self.log_dir_train=copy.deepcopy(log_dir)+'_'+'train'
#        self.log_dir_val=copy.deepcopy(log_dir)+'_'+'val'

        self.log_dir=log_dir
        if 1==1:
          if not os.path.exists(self.log_dir):
              os.mkdir(self.log_dir)
#          os.mkdir(self.log_dir_val)

          with open(self.log_dir+'/log_train.txt','a') as f_train:
                f_train.write('Epoch    Iter     Loss     MPJPE\n')
          with open(self.log_dir+'/log_val.txt','a') as f_val:
                f_val.write('Epoch    Iter    Loss    MPJPE\n')

    def write(self, txt,split):
        if split=='train':
            with open(self.log_dir+'/log_train.txt','a') as f_train:
                f_train.write(txt)

        if split=='val':
            with open(self.log_dir+'/log_val.txt','a') as f_val:
                f_val.write(txt)

## ref.py

#nJoints = 16 # H3.6M
# nJoints = 25 # NTU
nPAFs = 30
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3],
             [10, 15], [11, 14], [12, 13]]

# edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]

edges = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], 
         [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13], [14,15], [15,1], [1,16]]

nJoints = 17

dataset_name=['mpii']

# This corresponds to H36m and MPII
h36mImgSize = 224
outputRes = 64
inputRes = 256

# This corresponds to CMU
cmuzsize=384
cmuPadImgSize = 256
cmuImgSizew=384
cmuImgSizeh=216
cmuoutputResw=96
cmuoutputResh = 54

# This corresponds to coco
cocoSizeh = 240
cocoSizew = 320

# Sizes after padding
padSizew = 256
padSizeh = 256
maxPeople = 40
eps = 1e-6


datasets=1

momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8
root=6

scale = 0.25
rotate = 30
hmGauss = 1
hmGaussInp = 20
shiftPX = 50
disturb = 10

PCKthresh = 0.5

inp_img_size = 256
ntuDataDir = '/mnt/data/poorvi/new/skeletonData'
h36mDataDir = 'h36m_embeds'
dataDir = '/home/safeer/datasets/'
mpiiImgDir = '/mnt/data/CapsPose/data/images'
mpiiDataDir = '/mnt/data/CapsPose/data/annotations'
h36mImgDir = '/home/rishabh/CapsPose/data/h36m/images'
cocoImgDir = '/home/rishabh/CapsPose/data/coco/train2017'
cocoAnnot = '/home/rishabh/CapsPose/data/coco/annotations/person_keypoints_train2017.json'
cocoMaskDir = '/home/project/multipose/reporef/Pytorch_Realtime_Multi-Person_Pose_Estimation-master_COCO/masks/masklist_path'
cocoMaskDir_val='/home/project/multipose/reporef/Pytorch_Realtime_Multi-Person_Pose_Estimation-master_COCO/masks/masklist_path_val'
cocoAnnot_val='/home/project/multipose/MP3d/src_rishabh_coco/val_coco_annot.json'
cocoImgDir_val='/home/project/datasets/mscoco_multi/images/val2017'
cmuImgDir = '/home/project/multipose/panoptic-toolbox-master/scripts'
cmuAnnot = '/home/project/multipose/panoptic-toolbox-master/scripts/mixdata_train.json'
cmuAnnot_val='/home/project/multipose/panoptic-toolbox-master/scripts/mixdata_val.json'

visDir =['/home/project/multipose/src_2dtrain_coco/visualisations/h36m/','/home/project/multipose/src_rishabh_train_coco/visualisations/cmu/','/home/project/multipose/src_rishabh_train_coco/visualisations/coco/']




expDir = './exp'


cmuAnnotvar=''

nThreads = 4

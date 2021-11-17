import torch
import pickle
import torch.utils.data as data
import numpy as np
from h5py import File
import cv2
import os
import math
import torchvision

from data_loader.utils import load_annot_file, get_inputs_reg, load_camera_file
from utils.img import DrawGaussian
# from data_loader.augmentation import load_occluders, occlude_with_objects

train_subjects = [1, 5, 6, 7, 8]
test_subjects = [9, 11]


h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}

class H36M(data.Dataset):
    def __init__(self, opt, split, train_stats, K=4, allowed_subj_list_reg=None, allowed_cams=[1, 2, 3, 4], sil=False):
        print('==> initializing H36M {} data.'.format(split))
        print('data dir {}'.format(opt.data_dir))
        annot_file = os.path.join(opt.data_dir, 'annot_cam_' + ('train' if split == 'train' else 'test') + '.pickle')

        annot = load_annot_file(annot_file)
        self.sil = sil
        camera_path = os.path.join(opt.data_dir, 'cam_int_numpy.npy')
        self.cameras = np.load(camera_path)
        self.seg_dir = opt.seg_dir

        indi = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
        with np.load('./data/data_h36m.npz', allow_pickle=True) as data:
            self.kps = data['positions_2d'].item()
            self.meta = data['metadata'].item()

        subj_mask = np.zeros(annot['id'].shape[0], dtype='bool')
        for subj_id in allowed_subj_list_reg:
            subj_mask = np.logical_or(subj_mask, annot['subject'] == subj_id)

        valid_mask = subj_mask
        valid_ids = np.arange(annot['id'].shape[0])[valid_mask]

        for tag in annot.keys():
            annot[tag] = annot[tag][valid_ids]

        self.opt = opt
        self.annot = annot
        self.split = split
        self.allowed_subject_list = allowed_subj_list_reg
        self.n_joints = self.annot['joint_3d_rel'].shape[1]
        self.train_stats = train_stats
        self.get_rigidity(2, 6, thresh=0.05)


        self.multi_view_ids = self.get_multi_view_ids()  # no frames X no cams contains ids from annot
        # import pdb; pdb.set_trace()
        self.frame_chunks = self.get_multi_view_frame_chunks(K)

        if self.split == 'train':
            # self.occluders = load_occluders(pascal_voc_root_path=self.opt.pascal_voc_dir)
            self.occluders = None
        else:
            self.occluders = None

        self.img_mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)  # img net mean
        self.img_std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)  # img net var

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomAffine(10, (0.00, 0.00), (0.9, 1.1)),
            torchvision.transforms.ToTensor()
            ])

        # points 3d normalization
        pts_3d = np.copy(annot['joint_3d_rel'])
        pts_3d = pts_3d.reshape(pts_3d.shape[0], self.n_joints * 3)

        if 'mean_3d' not in self.train_stats.keys():
            print('Computing Mean and Std')
            self.mean_3d, self.std_3d = self.compute_mean_std(d=3)
            self.train_stats['mean_3d'] = self.mean_3d
            self.train_stats['std_3d'] = self.std_3d
            self.mean_2d, self.std_2d = self.compute_mean_std(d=2)
            self.train_stats['mean_2d'] = self.mean_2d
            self.train_stats['std_2d'] = self.std_2d
        else:
            print('Loading Mean and Std from args')
            self.mean_3d = self.train_stats['mean_3d']
            self.std_3d = self.train_stats['std_3d']

        eps = 1e-8
        pts_3d_norm = np.divide(pts_3d - self.mean_3d, self.std_3d + eps)

        self.annot['joint_3d_normalized'] = pts_3d_norm
        # self.n_samples = self.annot['id'].shape[0]
        self.n_samples = self.frame_chunks.shape[0]

        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))

    def get_rigidity(self, idx1, idx2, thresh):
        j3d = self.annot['joint_3d_rel']
        j1 = j3d[:, idx1, :]
        j2 = j3d[:, idx2, :]
        diff = np.sqrt(np.sum((j1 - j2)**2, 1))
        std = np.std(diff)
        dist = np.mean(diff)
        count = np.sum((diff > (1+thresh)*dist)*1.0) + np.sum((diff < (1-thresh)*dist)*1.0)

        return std, dist, (j1.shape[0] - count) / j1.shape[0]

    def compute_mean_std(self, d):
        if d == 3:
            pts_3d = np.copy(self.annot['joint_3d_rel'])
            pts_3d = pts_3d.reshape(pts_3d.shape[0], self.n_joints * 3)
        elif d == 2:
            pts_3d = np.copy(self.annot['joint_2d'])
            pts_3d = pts_3d.reshape(pts_3d.shape[0], self.n_joints * 2)

        mean_3d = np.mean(pts_3d, axis=0)
        std_3d = np.std(pts_3d, axis=0)

        return mean_3d, std_3d

    def get_multi_view_ids(self):
        no_cams = 4
        no_frames = self.annot['id'].shape[0]
        no_unq_frames = no_frames // no_cams
        multi_view_ids = np.zeros((no_unq_frames, no_cams), dtype='int32')

        for cam_id in range(1, no_cams+1):
            cam_mask = (self.annot['camera'] == cam_id)
            multi_view_ids[:, cam_id-1] = np.arange(0, no_frames)[cam_mask]

        return multi_view_ids

    def get_multi_view_frame_chunks(self, k=5):
        no_unq_frames = self.multi_view_ids.shape[0]
        no_frames = self.annot['id'].shape[0]
        frame_chunks = np.zeros((no_unq_frames*k), dtype='int32')

        shuffler_mv = np.arange(no_unq_frames)
        shuffler_basic = np.random.permutation(no_frames)

        for i in range(no_unq_frames):
            frame_chunks[i*k:i*k+4] = self.multi_view_ids[shuffler_mv[i], :]
            frame_chunks[i*k+4:(i+1)*k] = shuffler_basic[(i)*(k-4):(i+1)*(k-4)]

        return frame_chunks

    def __getitem__(self, index):

        annot_id = self.frame_chunks[index]
        id_reg, img_reg, pose_reg, pose_un_reg, pose_2d, subj_reg, unq_id_reg, cam_id, root, bbox = get_inputs_reg(
            self.opt, annot_id, self.annot, self.img_mean, self.img_std, self.occluders)

        
        p2d = self.kps
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        abs_pose_2d_x = (pose_2d[:, 0:1] * w / 224) + bbox[0]
        abs_pose_2d_y = (pose_2d[:, 1:2] * h / 224) + bbox[1]
        abs_pose = np.concatenate((abs_pose_2d_x, abs_pose_2d_y), -1)
        pose_2d = pose_2d - pose_2d[6:7,:]
        
        act = self.annot['action'][annot_id]
        subact = self.annot['subaction'][annot_id]
        camera = self.annot['camera'][annot_id]
        idx = self.annot['id'][annot_id]

        if self.sil:
            dir_name = f's_{subj_reg:02d}_act_{act:02d}_subact_{subact:02d}_ca_{camera:02d}'
            mask_path = os.path.join(self.seg_dir, dir_name, f'{dir_name}_{idx:06d}.jpg')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask[:1000, :1000]
        else:
            mask = np.zeros((24, 24, 3))

        # img_reg = img_reg.transpose(2,0,1) # HWC --> CHW

        # out = np.zeros((self.opt.n_joints, self.opt.output_res, self.opt.output_res))
        # for i in range(self.opt.n_joints):
        #     # pt = Transform(pts[i], c, s, r, self.opt.outputRes)
        #     pt = pose_2d[i] * 64 / 4 / 56
        #     out[i] = DrawGaussian(out[i], pt, self.opt.hm_gauss)
        meta = dict()
        meta['mask'] = mask
        meta['pose'] = pose_reg
        meta['abs_pose_2d'] = abs_pose
        meta['pose_un'] = pose_un_reg
        meta['pose_2d'] = pose_2d
        meta['mean_3d'] = self.mean_3d
        meta['std_3d'] = self.std_3d
        # meta['mean_2d'] = self.mean_2d
        # meta['std_2d'] = self.std_2d
        meta['annot_id'] = id_reg
        meta['subj'] = subj_reg
        meta['cam_id'] = cam_id
        meta['root'] = root
        meta['bbox'] = bbox*1.0
        meta['cameras'] = self.cameras[subj_reg-1, cam_id-1]
        meta['orientation'] = torch.tensor(h36m_cameras_extrinsic_params[f'S{subj_reg}'][cam_id-1]['orientation'])
        meta['translation'] = torch.tensor(h36m_cameras_extrinsic_params[f'S{subj_reg}'][cam_id-1]['translation']) 
        # if self.transforms and self.split=='train':
        #     img_reg = self.transforms(torch.from_numpy(img_reg))
        return meta

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    import pdb; pdb.set_trace()
    dataset_train = H36M(cfg, split='train',
                    train_stats={}, K=4,
                    allowed_subj_list_reg=[1,5,7,8])

    dataset_val = H36M(cfg, split='val',
                    train_stats={}, K=4,
                    allowed_subj_list_reg=[9,11])

    data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, batch_size=K,
                    shuffle=False, num_workers=4)

    for i, meta in enumerate(data_loader_train):
        pose = meta['pose_2d']
        import pdb; pdb.set_trace()
        pose_3d = meta['pose_un']

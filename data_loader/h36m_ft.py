import torch
import torch.utils.data as data
import numpy as np
from h5py import File
import cv2
import os
import math
import torchvision

from data_loader.utils import load_annot_file, get_inputs_reg, load_camera_file
from data_loader.utils import process_vp3d
from utils.img import DrawGaussian
from data_loader.camera import world_to_camera_rep as wtc
# from data_loader.augmentation import load_occluders, occlude_with_objects

train_subjects = [1, 5, 6, 7, 8]
test_subjects = [9, 11]

int_matrix = [[
        [1145, 0, -512.54],
        [0, 1143, -515.5],
        [0, 0, 1]
        ],
        [[1149, 0, -508,8],
        [0, 1147, -508.8],
        [0, 0, 1]],
        [[1149.14, 0, -519],
        [0, 1148, -501],
        [0, 0, 1]],
        [[1145, 0, -514],
        [0, 1144, -501],
        [0, 0, 1]]
    ]


h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]


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
    'S3': [
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
    'S4': [
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
    def __init__(self, opt, split, train_stats, K=4, allowed_subj_list_reg=None, allowed_cams=[1, 2, 3, 4]):
        print('==> initializing H36M {} data.'.format(split))
        print('data dir {}'.format(opt.data_dir))
        annot_file = os.path.join(opt.data_dir, 'annot_cam_' + ('train' if split == 'train' else 'test') + '.pickle')


        annot = load_annot_file(annot_file)
        camera_path = os.path.join(opt.data_dir, 'cam_int_numpy.npy')
        self.cameras = np.load(camera_path)

        self.annot_noisy = process_vp3d(opt.data_dir, split)
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
        self.frame_chunks = self.get_multi_view_frame_chunks(K)
        self.frame_chunks = self.frame_chunks.reshape(-1, 4)

        # points 3d normalization
        pts_3d = np.copy(annot['joint_3d_rel'])
        pts_3d = pts_3d.reshape(pts_3d.shape[0], self.n_joints * 3) 

        eps = 1e-8
        self.n_samples = len(self.annot_noisy['id'])
        # self.n_samples = self.frame_chunks.shape[0]

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

        # annot_ids = self.frame_chunks[index]
        p2d = np.zeros((4, 16, 2))
        p3d = np.zeros((4, 16, 3))
        sub = []
        cam = []
        roots = np.zeros((4, 3))
        bboxs = np.zeros((4,4))
        ori = torch.zeros((4, 4))
        trans = torch.zeros((4,3))
        p2d_noisy = self.annot_noisy['pose_2d'][index]
        id_reg = self.annot_noisy['id'][index]
        p3d_gt = torch.from_numpy(self.annot_noisy['pose_3d'][index]) * 1000
        subj_reg = self.annot_noisy['subject'][index]
        for i in range(4):
            # _, _, pose_3d, pose_2d, subj_reg, _, cam_id, root, bbox = get_inputs_reg(
            #                         self.opt, annot_ids[i], self.annot)
            # p2d[i] = pose_2d - pose_2d[6:7]
            # sub.append(subj_reg)
            cam_id = i + 1
            cam.append(cam_id)
            # roots[i] = root
            ori[i] = torch.Tensor(h36m_cameras_extrinsic_params[f'S{subj_reg}'][cam_id-1]['orientation'])
            trans[i] = torch.Tensor(h36m_cameras_extrinsic_params[f'S{subj_reg}'][cam_id-1]['translation']) 
            # bboxs[i] = bbox
            p3d[i] = wtc(p3d_gt, ori[i], trans[i])
            # w = bbox[2] - bbox[0]
            # h = bbox[3] - bbox[1]
            mins = p2d_noisy[i].min(0)
            maxs = p2d_noisy[i].max(0)
            w = maxs[0] - mins[0]
            h = maxs[1] - mins[1]
            # p2d_noisy[i] = (p2d_noisy[i] - bbox[:2]) * 224  / [w, h]
            p2d_noisy[i] = (p2d_noisy[i] - mins) / [w, h]
            p2d_noisy[i] = p2d_noisy[i] - p2d_noisy[i, 6:7]


        p3d = p3d - p3d[:, 6:7, :]
        # pose_2d = pose_2d - pose_2d[6:7,:]

        # img_reg = img_reg.transpose(2,0,1) # HWC --> CHW

        # out = np.zeros((self.opt.n_joints, self.opt.output_res, self.opt.output_res))
        # for i in range(self.opt.n_joints):
        #     # pt = Transform(pts[i], c, s, r, self.opt.outputRes)
        #     pt = pose_2d[i] * 64 / 4 / 56
        #     out[i] = DrawGaussian(out[i], pt, self.opt.hm_gauss)
        meta = dict()
        meta['pose_un'] = p3d
        meta['pose_noisy'] = p2d_noisy
        # meta['pose_2d'] = p2d
        meta['annot_id'] = id_reg
        meta['subj'] = subj_reg
        meta['cam_id'] = cam_id
        # meta['root'] = root
        # meta['cameras'] = self.cameras[subj_reg-1, cam_id-1]
        # meta['bbox'] = bboxs
        meta['orientation'] = ori
        meta['translation'] = trans
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


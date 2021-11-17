import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
              [6, 8], [8, 9]]

mpii_edges_15 = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [9, 10],
                 [10, 11], [11, 7], [7, 12], [12, 13], [13, 14], [6, 7], [7, 8]]

# def show_2d(img, points, c, edges):
#     num_joints = points.shape[0]
#     points = ((points.reshape(num_joints, -1))).astype(np.int32)
#     for j in range(num_joints):
#         cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
#     for e in edges:
#         if points[e].min() > 0:
#             cv2.line(img, (points[e[0], 0], points[e[0], 1]),
#                      (points[e[1], 0], points[e[1], 1]), c, 2)
#     return img


class Debugger(object):
    def __init__(self, edges=mpii_edges):

        self.plt = plt

        oo = 1e10
        self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
        self.xmin, self.ymin, self.zmin = oo, oo, oo
        self.imgs = {}
        self.pts_3d = {}
        self.edges = edges
        self.img_count = 0
        self.pts_3d_count = 0

    def add_img(self, img):
        self.img_count = self.img_count + 1
        self.imgs[self.img_count] = img.copy()

    def add_pt_3d(self, pt_3d):
        pt_3d = pt_3d.reshape(-1, 3)
        self.pts_3d_count = self.pts_3d_count + 1

        self.pts_3d[self.pts_3d_count] = pt_3d

    def plot_3D(self, ax, pt_3d):
        self.xmax = 1000
        self.ymax = 1000
        self.zmax = 1000
        self.xmin = -1000
        self.ymin = -1000
        self.zmin = -1000

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        # ax.set_zlim(self.zmin, self.zmax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # ax.grid(True)

        c = 'b'
        marker = 'o'

        x, y, z = np.zeros((3, pt_3d.shape[0]))

        for j in range(pt_3d.shape[0]):
            x[j] = pt_3d[j, 0].copy()
            y[j] = pt_3d[j, 1].copy()  # pt_3d[j, 2].copy()
            z[j] = pt_3d[j, 2].copy()

        ax.scatter(x, y, s=20, c=c, marker=marker)
        for e in self.edges:
            ax.plot(x[e], y[e], c=c)

        # ax.view_init(elev=120, azim=90)

    def show(self):

        ax = plt.subplot(121)
        plt.imshow(self.imgs[1])

        ax_1 = plt.subplot(122)  # , projection='3d')
        self.plot_3D(ax_1, self.pts_3d[1])

        # ax_2 = plt.subplot(133, projection='3d')
        # self.plot_3D(ax_2, self.pts_3d[2])

        plt.show(block=True)


def show_img_pose(imgs, poses, n_joints=16):

    if imgs is not None and imgs.ndim < 4:
        imgs = imgs.reshape(1, imgs.shape[0], imgs.shape[1], imgs.shape[2])

    poses = poses.reshape(-1, n_joints, 3)

    # assert imgs.shape[0] == poses_glb.shape[0]

    for i in range(poses.shape[0]):
        if n_joints == 15:
            dbg = Debugger(edges=mpii_edges_15)
        else:
            dbg = Debugger(edges=mpii_edges)

        dbg.add_img(imgs[i])

        dbg.add_pt_3d(poses[i])

        dbg.show()


class DebuggerBatch(object):
    def __init__(self):

        self.plt = plt

    def show_image_stacks(self, imgs_a, imgs_p):

        assert imgs_a.shape[0] == imgs_p.shape[0]

        n_imgs = min(4, imgs_a.shape[0])

        for i in range(n_imgs):
            plt.subplot(n_imgs, 2, i*2+1)
            plt.imshow(imgs_a[i])

            plt.subplot(n_imgs, 2, i*2+2)
            plt.imshow(imgs_p[i])

        plt.show(block=True)

        plt.close()

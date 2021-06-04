import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
# from models.animoji_networks import AnimojiEyeNet, AnimojiMouthNet


class TensorTransformer():

    def __init__(self):
        super(TensorTransformer, self).__init__()

    def get_rotation_mtx_2d(self, roll, scale, centor, t):
        """
        roll : [N]
        scale : [N]
        centor: [N, 2]
        t: [N, 2]
        """
        raise NotImplementedError
        alpha = scale * torch.cos(roll)
        beta = scale * torch.sin(roll)

        tmp1 = (1 - alpha) * centor[:, 0] - beta * centor[:, 1] + t[:, 0]
        tmp2 = beta * centor[:, 0] + (1 - alpha) * centor[:, 1] + t[:, 1]

        mtx = torch.cat([alpha, beta, tmp1, -beta, alpha, tmp2], dim=1)
        mtx = mtx.view(-1, 2, 3)
        return mtx

    def build_identical_grid(self, source_size):
        """
        source_size: (w, h)
        target_size: (w, h)
        """
        w_s, h_s = source_size

        # [Ht, Wt, 1]
        x_grid = torch.arange(0, w_s).unsqueeze(0).repeat(h_s, 1).unsqueeze(-1)
        x_grid = (x_grid - (w_s - 1) / 2.) * (2. / w_s)
        # [Wt, Ht, 1]
        y_grid = torch.arange(0, h_s).unsqueeze(0).repeat(w_s, 1).unsqueeze(-1)
        y_grid = (y_grid - (h_s - 1) / 2.) * (2. / h_s)

        full = torch.cat([x_grid, y_grid.transpose(1, 0)],
                         dim=2).unsqueeze(0)  # [1, T, T, 2]
        return full  # F.affine_grid(torch.tensor([[[1, 0, 0], [0, 1, 0]]]).float(), [1, 1, h_s, w_s])

    def build_central_crop_grid(self, source_size, target_size):
        """
        source_size: (w, h)
        target_size: (w, h)
        """
        raise NotImplementedError
        w_s, h_s = source_size
        w_t, h_t = target_size
        k_w = float(w_t) / float(w_s)
        k_h = float(h_t) / float(h_s)

        # [Ht, Wt, 1]
        x_grid = torch.linspace(-k_w, k_w,
                                w_t).unsqueeze(0).repeat(h_t, 1).unsqueeze(-1)
        # [Wt, Ht, 1]
        y_grid = torch.linspace(-k_h, k_h,
                                h_t).unsqueeze(0).repeat(w_t, 1).unsqueeze(-1)

        full = torch.cat([x_grid, y_grid.transpose(1, 0)],
                         dim=2).unsqueeze(0)  # [1, T, T, 2]
        return full

    def build_resize_grid(self, source_size, target_size):
        """
        source_size: (w, h)
        target_size: (w, h)
        """
        w_s, h_s = source_size
        w_t, h_t = target_size
        k_w = float(w_t) / float(w_s)
        k_h = float(h_t) / float(h_s)

        # [Ht, Wt, 1]
        x_grid = torch.arange(0, w_t).unsqueeze(0).repeat(h_t, 1).unsqueeze(-1)
        x_grid = (x_grid - (w_t - 1) / 2.) * (2. / w_t)
        # [Wt, Ht, 1]
        y_grid = torch.arange(0, h_t).unsqueeze(0).repeat(w_t, 1).unsqueeze(-1)
        y_grid = (y_grid - (h_t - 1) / 2.) * (2. / h_t)

        full = torch.cat([x_grid, y_grid.transpose(1, 0)],
                         dim=2).unsqueeze(0)  # [1, T, T, 2]
        return full

    def build_crop_resize_grid(self, source_size, target_size, centors, crop_size):
        """
        source_size: (w, h)
        target_size: (w, h)
        centors: [N, 2]
        crop_size: [N, 2]
        """
        N = centors.size(0)
        central_grid = self.build_resize_grid(source_size, target_size).repeat(
            N, 1, 1, 1).to(crop_size.device)  # [N, T, T, 2]

        crop_ratio = crop_size.clone()
        crop_ratio[:, 0] /= float(source_size[0])
        crop_ratio[:, 1] /= float(source_size[1])

        scale_grid = central_grid * crop_ratio.view(N, 1, 1, 2)  # [N, T, T, 2]

        shifter = centors.clone()
        shifter[:, 0] = (shifter[:, 0] - source_size[0] /
                         2.0) / float(source_size[0]) * 2
        shifter[:, 1] = (shifter[:, 1] - source_size[1] / 2.0) / \
            float(source_size[1]) * 2  # [N, 2]
        shifted_grid = scale_grid + shifter.view(N, 1, 1, 2)
        return shifted_grid

    def compute_rotation_matrix(self, angle, source_size=None):
        """
        args:
            - angle : tensor of shape (N, ), positive means anticlockwise

        return:
            - rotation_matrix: tensor of shape (N, 2, 2)
            newp = oldp.dot(rotation_matrix)
        """
        _cos = torch.cos(angle).unsqueeze(1)
        _sin = torch.sin(angle).unsqueeze(1)
        if source_size is None:
            rotation_matrix = torch.cat(
                [_cos, -_sin, _sin, _cos], dim=1).view(-1, 2, 2)
        else:
            w, h = source_size
            rotation_matrix = torch.cat(
                [_cos, -_sin*w/h, _sin*h/w, _cos], dim=1).view(-1, 2, 2)
        return rotation_matrix

    def build_central_rotation_grid(self, source_size, rotation_matrix):
        """
        args:
            - source_size : (w, h)
            - rotation_matrix : tensor of shape (N, 2, 2)
        """
        N = rotation_matrix.size(0)
        iden_grid = self.build_identical_grid(source_size).to(
            rotation_matrix.device).repeat(N, 1, 1, 1)  # NWH2
        rotation_matrix = self.reverse_rotation_matrix(rotation_matrix)  # N22
        rotation_grid = torch.matmul(iden_grid, rotation_matrix.unsqueeze(1))
        return rotation_grid

    def build_translation_grid(self, source_size, translation):
        """
        args:
            - source_size : (w, h)
            - translation : tensor of shape (N, 2), positive means rightdown
        """
        N = translation.size(0)
        iden_grid = self.build_identical_grid(source_size).to(
            translation.device).repeat(N, 1, 1, 1)  # NWH2

        shifter = translation.clone()
        shifter[:, 0] = shifter[:, 0] / float(source_size[0]) * 2
        shifter[:, 1] = shifter[:, 1] / float(source_size[1]) * 2  # [N, 2]

        translation_grid = iden_grid - shifter.view(N, 1, 1, 2)
        return translation_grid

    def build_rotation_grid(self, source_size, rotate_centors, rotate_angles):
        """
        args:
            - source_size : (w, h)
            - rotation_matrix : tensor of shape (N, 2, 2)
        """
        N = rotate_centors.size(0)
        rotation_matrix = self.compute_rotation_matrix(-rotate_angles, source_size)
        iden_grid = self.build_identical_grid(source_size).to(
            rotation_matrix.device).repeat(N, 1, 1, 1)  # NWH2

        shifter = rotate_centors.clone()
        shifter[:, 0] = (shifter[:, 0] - source_size[0] /
                         2.0) / float(source_size[0]) * 2
        shifter[:, 1] = (shifter[:, 1] - source_size[1] /
                         2.0) / float(source_size[1]) * 2

        # coordinate transformation
        shifter_grid = iden_grid - shifter.view(N, 1, 1, 2)
        rotation_grid = torch.matmul(
            shifter_grid, rotation_matrix.unsqueeze(1))
        rotation_grid = rotation_grid + shifter.view(N, 1, 1, 2)

        return rotation_grid

    def rotate_points(self, points, rotate_centors, rotate_angles):
        """
        args:
            - points : [N, P, 2], P is number of points for each image
            - rotate_centors : [N, 2]
            - rotate_angles : [N, ]

        return:
            - rotated points

        """
        rotation_matrix = self.compute_rotation_matrix(rotate_angles)  # [N, 2, 2]

        temp_points = points - rotate_centors.unsqueeze(1)
        temp_points = torch.matmul(temp_points, rotation_matrix)
        rotated_points = temp_points + rotate_centors.unsqueeze(1)

        return rotated_points

    def rotate_face(self, x, rotate_centors, rotate_angles):
        """
        args:
            - x : NCHW
            - rotate_centors: [N, 2]
            - rotate_angles: [N, ]

        return:
            - out_x : of the same shape as x
        """
        h, w = x.size(2), x.size(3)

        rotation_grid = self.build_rotation_grid(
            (w, h), rotate_centors, rotate_angles)
        rotated_x = F.grid_sample(x, rotation_grid, align_corners=False)

        return rotated_x

    def crop_resize_face(self, x, target_size, centors, crop_sizes):
        h, w = x.size(2), x.size(3)

        crop_grid = self.build_crop_resize_grid(
            (w, h), target_size, centors, crop_sizes)
        cropped_x = F.grid_sample(x, crop_grid, align_corners=False)
        return cropped_x

    def rotate_then_crop_resize_face(self, x, rotate_centors, rotate_angles, target_size, crop_centors, crop_sizes):
        rotated_x = self.rotate_face(x, rotate_centors, rotate_angles)
        cropped_x = self.crop_resize_face(
            rotated_x, target_size, crop_centors, crop_sizes)
        return cropped_x

######################

    def get_crop_box_from_lds(self, lds, part):
        # lds: [N, 101, 2]
        assert part in ["mouth", "eye"]
        length = torch.max(lds[:, :, 0], dim=1)[0] - \
            torch.min(lds[:, :, 0], dim=1)[0]
        crop_size = None
        centor = None

        if part == "mouth":
            scale = (-0.3, -0.35, 0.3, 0.25)
            centor = (torch.max(lds[:, 75:95, :], dim=1)[
                      0] + torch.min(lds[:, 75:95, :], dim=1)[0]) * 0.5
            centor[:, 0] += (scale[0] + scale[2]) / 2.0 * length / 2.0
            centor[:, 1] += (scale[1] + scale[3]) / 2.0 * length / 2.0

            width = (1 + abs(scale[2] - scale[0])) * length / 2.0
            height = (1 + abs(scale[3] - scale[1])) * length / 2.0
            crop_size = torch.cat(
                [width.unsqueeze(1), height.unsqueeze(1)], dim=1)
        elif part == "eye":
            scale = (-0.5, -0.2, 0.5, -0.2)
            centor = lds[:, 97, :]
            centor[:, 0] += (scale[0] + scale[2]) / 2.0 * length / 2.0
            centor[:, 1] += (scale[1] + scale[3]) / 2.0 * length / 2.0

            width = (1 + abs(scale[2] - scale[0])) * length / 2.0
            height = (1 + abs(scale[3] - scale[1])) * length / 2.0
            crop_size = torch.cat(
                [width.unsqueeze(1), height.unsqueeze(1)], dim=1)
        return centor, crop_size

    def crop(self, x, lds, part):
        """
        x: NCHW, BGR, [-1, 1]
        lds: [N, 101, 2]
        part: "mouth", "eye"
        source_size: (w, h)
        """
        assert part in ["mouth", "eye"]
        source_size = (x.size(3), x.size(2))
        target_size = None
        if part == "mouth":
            target_size = (96, 96)
        elif part == "eye":
            target_size = (96, 48)
        centor, crop_size = self.get_crop_box_from_lds(lds, part)
        # print(centor, crop_size)
        crop_grid = self.build_crop_grid(
            source_size, target_size, centor, crop_size)
        cropped_x = F.grid_sample(
            x * 0.5 + 0.5, crop_grid, align_corners=False)
        cropped_x = (cropped_x - 0.5) / 0.5
        return cropped_x

    def transform(self, x):
        """
        x is NCHW, BGR, [-1, 1]

        return: grey image, NCHW, Y, [-1, 1]
        """
        x = x * 0.5 + 0.5
        x_grey = x[:, [2], :, :] * 299 / 1000 + x[:, [1], :, :] * \
            587 / 1000 + x[:, [0], :, :] * 114 / 1000
        x_grey = (x_grey - 0.5) / 0.5
        return x_grey
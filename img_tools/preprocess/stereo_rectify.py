import cv2
import os
import shutil
from os import listdir
from os.path import join, split, splitext
import json
import numpy as np


def rectify(frame_parameter_file, left_raw, right_raw):
    with open(frame_parameter_file) as para_json_file:
        data = json.load(para_json_file)
        camera_parameter = data['camera-calibration']

        l_camera_matrix = np.array(camera_parameter['KL']) #左内参矩阵
        r_camera_matrix = np.array(camera_parameter['KR'])#由内参
        l_dist_coeff = np.array(camera_parameter['DL']) #左畸变系数（径向和切向）
        r_dist_coeff = np.array(camera_parameter['DR']) #右畸变系数（径向和切向）
        rotation = np.array(camera_parameter['R'])
        translation = np.reshape(np.array(camera_parameter['T']),(3,1))

        img_size = left_raw.shape[1::-1]
        print(img_size)

        #？
        R1, R2, P1, P2, Q, ROI_l, ROI_r = cv2.stereoRectify(l_camera_matrix, l_dist_coeff, r_camera_matrix, r_dist_coeff, img_size, rotation, translation) 
        """
        R1: 3x3的校正旋转矩阵，将未校正的左目图像坐标变换到校正后的左目图像坐标
        R2: 3x3的校正旋转矩阵，将未校正的右目图像坐标变换到校正后的右目图像坐标
        P1: 3x4的投影矩阵，将校正后的左目图像坐标投影到图像平面坐标
        P2: 3x4的投影矩阵，将校正后的右目图像坐标投影到图像平面坐标
        Q: 4x4的视差深度映射矩阵，用于从视差图计算三维点坐标
        ROI_l: 一个最多地包含有效像素的长方形，用于裁剪校正后的左目图像
        ROI_r: 一个最多地包含有效像素的长方形，用于裁剪校正后的右目图像
        投影矩阵3*4是一个用于将三维空间中的点投影到二维平面上的矩阵，它可以表示为12：

        
        """
        mapLx, mapLy = cv2.initUndistortRectifyMap(l_camera_matrix, l_dist_coeff, R1, P1, img_size, cv2.CV_32F)
        mapRx, mapRy = cv2.initUndistortRectifyMap(r_camera_matrix, r_dist_coeff, R2, P2, img_size, cv2.CV_32F)

        print('map shape:' + str(mapLx.shape) + str(mapLy.shape))

        left_finalpass = cv2.remap(left_raw, mapLx, mapLy, cv2.INTER_LINEAR)
        right_finalpass = cv2.remap(right_raw,mapRx, mapRy, cv2.INTER_LINEAR)

        return left_finalpass, right_finalpass, Q


def save_Q(Q, reprojection_file):

    data = {'reprojection-matrix': Q.tolist()}
    with open(reprojection_file, 'w') as outfile:
        json.dump(data, outfile, separators=(',', ':'), sort_keys=True, indent=4)

def save_finalpass(left_img, right_img, left_file, right_file):
    cv2.imwrite(left_file, left_img)
    cv2.imwrite(right_file, right_img)


def stereo_rectify(path):

    rootpath = path

    keyframe_list = [join(rootpath, kf) for kf in listdir(rootpath) if ('keyframe' in kf and 'ignore' not in kf)]
    for kf in keyframe_list:
        left_raw_filepath = join(rootpath, kf) + '/data/left'

        if not os.path.isdir(left_raw_filepath):
            continue

        right_raw_filepath = join(rootpath, kf) + '/data/right'
        frame_para_filepath = join(rootpath, kf) + '/data/frame_data'
        img_filelist = [sf for sf in listdir(left_raw_filepath) if '.png' in sf]
        for sf in img_filelist:
            # stereo rectify
            left_raw_file = join(left_raw_filepath, sf)
            right_raw_file = join(right_raw_filepath, sf)
            filename, ext = splitext(sf)
            frame_para_file = join(frame_para_filepath, filename + '.json')

            left_raw = cv2.imread(left_raw_file)
            right_raw = cv2.imread(right_raw_file)
            print(frame_para_file)
            left_finalpass, right_finalpass, Q = rectify(frame_para_file, left_raw, right_raw)

            if not os.path.isdir(join(rootpath, kf) + '/data/left_finalpass'):
                os.mkdir(join(rootpath, kf) + '/data/left_finalpass')
            if not os.path.isdir(join(rootpath, kf) + '/data/right_finalpass'):
                os.mkdir(join(rootpath, kf) + '/data/right_finalpass')
            if not os.path.isdir(join(rootpath, kf) + '/data/reprojection_data'):
                os.mkdir(join(rootpath, kf) + '/data/reprojection_data')

            # save final pass image and reprojection matrix
            left_finalpass_savefile = join(rootpath, kf) + '/data/left_finalpass/' + sf
            right_finalpass_savefile = join(rootpath, kf) + '/data/right_finalpass/' + sf
            reprojection_file = join(rootpath, kf) + '/data/reprojection_data/' + filename + '.json'

            save_finalpass(left_finalpass, right_finalpass, left_finalpass_savefile, right_finalpass_savefile)
            save_Q(Q, reprojection_file)
        
        shutil.rmtree(join(rootpath, kf) + '/data/left')
        shutil.rmtree(join(rootpath, kf) + '/data/right')

if __name__ == '__stereo_rectify__':
    path = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset3'
    stereo_rectify(path)
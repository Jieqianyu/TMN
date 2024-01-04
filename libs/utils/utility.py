import numpy as np
import math

import torch
import os
import shutil
import cv2
import random
from time import time
import sys
import pandas as pd

from PIL import Image
from options import OPTION as opt
from ..dataset.data import ROOT_DAVIS, convert_mask
from libs.davis2017.evaluation import DAVISEvaluation


def davis2017_eval(results_path, davis_path=ROOT_DAVIS, task='semi-supervised', set='val', version='2017'):
    time_start = time()
    print(f'Evaluating sequences for the {task} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=set, version=version)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']
    
    # Path 
    csv_name_global = f'global_results-{version}{set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{version}{set}.csv'
    csv_name_global_path = os.path.join(results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'a') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'a') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    
    return final_mean

def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint', freq=1):
    if epoch >= 100 or epoch % 10==0:
        filepath = os.path.join(checkpoint, filename + '_{}'.format(str(epoch)) + '.pth.tar')
    else:
        filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    print('==> save model at {}'.format(filepath))

    if is_best:
        cpy_file = os.path.join(checkpoint, filename+'_model_best.pth.tar')
        shutil.copyfile(filepath, cpy_file)
        print('==> save best model at {}'.format(cpy_file))

def write_mask(mask, info, opt):

    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    directory = os.path.join(opt.results)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(mask[0].shape[0])]
    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        boxes = get_bbox(rescale_mask)
        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t]) 
        if opt.save_indexed_format:
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')
        else:
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max()+1):
                seg[rescale_mask==k, :] = info['palette'][(k*3):(k+1)*3]
            if opt.valset == 'DAVIS17' or opt.valset == 'DAVIS16':
                inp_img = cv2.imread(os.path.join(ROOT_DAVIS, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            else:
                raise NameError
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            for i, box in enumerate(boxes):
                if box is not None:
                    b_x, b_y, b_w, b_h = box 
                    cv2.rectangle(im, (b_x, b_y), (b_x+b_w, b_y+b_h), colors[i], 2)
            cv2.imwrite(os.path.join(video, output_name), im)
        

def mask_iou(pred, target):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum(inter / union) / N

    return iou

def adjust_learning_rate(optimizer, epoch, opt):

    if epoch in opt.milestone:
        opt.learning_rate *= opt.gamma
        for pm in optimizer.param_groups:
            pm['lr'] = opt.learning_rate

def pointwise_dist(points1, points2):

    # compute the point-to-point distance matrix

    N, d = points1.shape
    M, _ = points2.shape

    p1_norm = torch.sum(points1**2, dim=1, keepdim=True).expand(N, M)
    p2_norm = torch.sum(points2**2, dim=1).unsqueeze(0).expand(N, M)
    cross = torch.matmul(points1, points2.permute(1, 0))

    dist = p1_norm - 2 * cross + p2_norm

    return dist

def furthest_point_sampling(points, npoints):

    """
    points: [N x d] torch.Tensor
    npoints: int

    """
    
    old = 0
    output_idx = []
    output = []
    dist = pointwise_dist(points, points)
    fdist, fidx = torch.sort(dist, dim=1, descending=True)

    for i in range(npoints):
        fp = 0
        while fp < points.shape[0] and fidx[old, fp] in output_idx:
            fp += 1

        old = fidx[old, fp]
        output_idx.append(old)
        output.append(points[old])

    return torch.stack(output, dim=0)


def get_bbox(mask):
    max_obj = np.max(mask)
    mask_one_hot = convert_mask(mask, max_obj).transpose(2, 0, 1).astype(np.uint8)
    boxes = list()
    for i in range(1, max_obj+1):
        contours, _ = cv2.findContours(mask_one_hot[i,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            # target_contour = max(contours, key=lambda x: x.shape[0])
            target_contour = np.concatenate(contours, axis=0)
            boxes.append(np.int0(cv2.boundingRect(target_contour))) # xywh
        else:
            boxes.append(None)
    return boxes


def mask_state(pred_mask, eps=1e-7):
    """
    param: pred_mask of size [N x H x W]
    """
    pred = pred_mask.clone().cpu().numpy()
    max_obj = pred.shape[0] - 1
    pred_b = convert_mask(pred.argmax(axis=0), max_obj).transpose(2, 0, 1).astype(np.uint8)

    conf_score = (pred * pred_b).sum(2).sum(1) / (pred_b.sum(2).sum(1) + eps)

    conc_score = np.zeros(conf_score.shape)
    for i in range(max_obj+1):
        contours, _ = cv2.findContours(pred_b[i,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 10:
            conc_score[i] = np.max(cnt_area) / sum(cnt_area)

    state_score = conf_score * conc_score

    return state_score
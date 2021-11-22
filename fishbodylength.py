from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from yolact_cpu import Yolact as Yolact_cpu
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import json
import sys
import os
from pathlib import Path
import pandas as pd 
import mmcv

import matplotlib.pyplot as plt
import cv2
import timeit
from skimage.morphology import skeletonize, thin
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import KFold
from maskLength import *
import csv
import joblib

NEWFISH_16_CLASSES = ["cp", "ALB", 'BET', 'BUM', 'DOL', 'LEC', 
					  'MLS', 'OIL', 'SBT', 'SFA', 'SKJ', 
					  'SSP', 'SWO', 'Shark', 'WAH', 'YFT']

c = ["black", "lightseagreen", "deepskyblue", "olivedrab", "gold","firebrick",
	 "yellowgreen","violet", "royalblue", "darkolivegreen", "steelblue",
	 "forestgreen", "seagreen", "indigo","darkorange", "darkblue"]

model_time = 0
process_time = 0
length_time = 0
mae_column = ["KFold", "Title", "ALB", 'BET', 'BUM', 'DOL', 'LEC', 
			  'MLS', 'OIL', 'SBT', 'SFA', 'SKJ', 
			  'SSP', 'SWO', 'Shark', 'WAH', 'YFT', "Average"]
pred_column = ['Image', 'Class', 'Length']
mae_list = []
SEED = 13
output_path = '/home/nas/Research_Group/Personal/HongYang/LengthResults/figures/yolact'

def length_extract(image_path=None, net=None):
	global model_time, process_time, length_time

	gt_length_list = []
	length_list = []
	gt_label_list = []
	label_list = []
	prog_bar = mmcv.ProgressBar(len(image_path))

	for image_id, p in enumerate(image_path): 
		length = 0
		path = Path(p)
		#print(path)

		# Yolact pred time-----------------------------
		model_time_start = timeit.default_timer()
		try:
			if args.cuda:
				img = torch.from_numpy(cv2.imread(p)).cuda().float()
			else:
				img = torch.from_numpy(cv2.imread(p)).float()
		except Exception as e:
			continue

		batch = FastBaseTransform()(img.unsqueeze(0))
		#print(batch.shape)
		#sys.exit()
		preds = net(batch)
		prog_bar.update()
		
		model_time_end = timeit.default_timer()
		model_time += (model_time_end - model_time_start)
		# ---------------------------------------------

		# Process time---------------------------------
		process_time_start = timeit.default_timer()
		h, w, _ = img.shape
		img_numpy = img.cpu().numpy().astype(np.uint8)

		t = postprocess(preds, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = args.score_threshold)

		idx = t[1].argsort(0, descending=True)[:args.top_k]
		classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]

		num_dets_to_consider = min(args.top_k, classes.shape[0])
		for j in range(num_dets_to_consider):
			if scores[j] < args.score_threshold:
				num_dets_to_consider = j
				break

		if num_dets_to_consider > 0:
			masks = masks[:num_dets_to_consider, :, :, None]

		process_time_end = timeit.default_timer()
		process_time += (process_time_end - process_time_start)
		# ---------------------------------------------

		# Load ground truth length from json
		json_path = 'data/LengthMeasure/labels/' + path.stem + '.json'
		with open(json_path) as f:
			gt_json = json.load(f)

		gt = gt_json['shapes'][0]
		(x1, y1), (x2, y2) = gt['points']
		gt_length = ((x1-x2)**2 + (y1-y2)**2)**0.5

		gt_length_list.append(gt_length)
		
		# get ground truth category from file name
		for idx, name in enumerate(NEWFISH_16_CLASSES):
			if idx == 0: continue
			if (str(path.stem).find(name) != -1):
				gt_label_tmp = idx
				gt_label_list.append(idx)
				break
		else:
			if (str(path.stem).find("BLZ") != -1):
				gt_label_tmp = 3
				gt_label_list.append(3)
			# Shark
			else:
				gt_label_tmp = 13
				gt_label_list.append(13)

		# Length time----------------------------------
		length_time_start = timeit.default_timer()
		for i, mask in enumerate(masks):
			if classes[i] == 0:
				continue
			mask = np.squeeze(mask)

			#length = mask_mixed(mask, classes[i])
			length = mask_elli(mask)
			
			length_list.append(length)
			label_list.append(classes[i])
			break
		else:
			length_list.append(0)
			label_list.append(gt_label_tmp)
		# ----------------------------------------------
		length_time_end = timeit.default_timer()
		length_time += (length_time_end - length_time_start)

	return np.array(gt_length_list), np.array(gt_label_list), np.array(length_list), np.array(label_list)
			
def train(gt_len, gt_label, pred_len, k='all'):
	uni_label = np.unique(gt_label)
	linear_dict = {}

	for i in uni_label:
		print(f'{NEWFISH_16_CLASSES[i]} start training')
		
		x = pred_len[np.where(gt_label == i)]
		y = gt_len[np.where(gt_label == i)]
		
		#model = LinearRegression()
		model = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=False))
		model.fit(x.reshape(-1,1), y)
		joblib.dump(model, f"./{args.output_folder}/{NEWFISH_16_CLASSES[i]}_model_new_{k}.joblib")

		linear_dict[NEWFISH_16_CLASSES[i]] = model
	
	return linear_dict

def plot_error_all(gt_length_list, length_list, label_list, title, linear_dict=None, k=None):
	print(f'\n{title}')

	mae_dict = {"KFold": k, "Title": title}
	pred_dict = {}
	mae_total = 0

	for i in np.unique(label_list):
		idx = np.where(label_list == i)
		mae, pred_list = plot_error_ind(gt_length_list[idx], length_list[idx], i, title, linear_dict)
		mae_dict[NEWFISH_16_CLASSES[i]] = mae
		pred_dict[NEWFISH_16_CLASSES[i]] = pred_list
		mae_total += mae

	fig, ax = plt.subplots(1)

	for i in np.unique(label_list):
		idx = np.where(label_list == i)
		ax.scatter(gt_length_list[idx], pred_dict[NEWFISH_16_CLASSES[i]], s=8, c=[c[i]], label=NEWFISH_16_CLASSES[i])
	
	#max_value = max(max(gt_length_list), max(length_list))
	max_value = 750
	#min_value = min(min(gt_length_list), min(length_list))
	min_value = 300

	ax.plot([min_value, max_value], 
		    [min_value, max_value], 
		    color=[0,0,0], label="y=x")
	
	order = [0,1,2,8,15,10,3,6,9,11,12,4,5,7,14,13] 
	handles,labels = ax.get_legend_handles_labels()
	ax.set_aspect('equal')

	handles = [handles[idx] for idx in order]
	labels = [labels[idx] for idx in order]

	#plt.title(f'Fish Body Length Estimation: ALL')
	plt.xlabel('Ground truth (pixel)', fontsize=14)
	plt.ylabel('Estimated (pixel)', fontsize=14)
	plt.xlim(min_value, max_value)
	plt.ylim(min_value, max_value)
	plt.xticks(np.arange(min_value, max_value+50, 50), fontsize=12)
	plt.yticks(np.arange(min_value, max_value+50, 50), fontsize=12)
	ax.legend(handles, labels, loc = "lower right", ncol=3, fontsize=8)
	fig.savefig(f'{args.output_folder}/length15_new_{title}_ALL_{k}.png', dpi=300)
	plt.close(fig)
	plt.clf()

	mae_dict["Average"] = mae_total / len(np.unique(label_list))
	print(f'Average: {mae_dict["Average"]:.2f}%')
	
	global mae_list
	mae_list.append(mae_dict)

def plot_error_ind(gt_length_list, length_list, label, title, linear_dict=None):

	if linear_dict:
		linear_model = linear_dict[NEWFISH_16_CLASSES[label]]
		length_list = linear_model.predict(length_list.reshape((-1,1)))

	abs_error_list = np.abs(length_list - gt_length_list)
	abs_error_perc_list = abs_error_list / gt_length_list
	mae = np.mean(abs_error_perc_list)*100
	
	#print(f'{NEWFISH_16_CLASSES[label]}: {mae:.2f}%')


	return mae, length_list

def kf_eval(all_path, net):
	gt_len, gt_label, pred_len, pred_label = length_extract(all_path, net)

	kf = KFold(n_splits=5)

	for k, (train_idx, test_idx) in enumerate(kf.split(gt_len)):
		train_gt_len, train_gt_label, train_pred_len, train_pred_label = gt_len[train_idx], gt_label[train_idx], pred_len[train_idx], pred_label[train_idx]
		test_gt_len, test_gt_label, test_pred_len, test_pred_label = gt_len[test_idx], gt_label[test_idx], pred_len[test_idx], pred_label[test_idx]

		linear_dict = train(train_gt_len, train_gt_label, train_pred_len, k)

		plot_error_all(train_gt_len, train_pred_len, train_gt_label, "train_rough", None, k)
		plot_error_all(train_gt_len, train_pred_len, train_pred_label, "train_linear", linear_dict, k)
		
		plot_error_all(test_gt_len, test_pred_len, test_gt_label, "test_rough", None, k)
		plot_error_all(test_gt_len, test_pred_len, test_pred_label, "test_linear", linear_dict, k)

def eval(image_path, net):
	# training data
	gt_len, gt_label, pred_len, pred_label = length_extract(image_path, net)
	linear_dict = train(gt_len, gt_label, pred_len)
	plot_error_all(gt_len, pred_len, gt_label, "image_rough")
	plot_error_all(gt_len, pred_len, pred_label, "image_linear", linear_dict)

def infer(image_path, net):
	global model_time, process_time, length_time

	result_list = []
	prog_bar = mmcv.ProgressBar(len(image_path))

	for image_id, p in enumerate(image_path): 
		length = 0

		# Yolact pred time-----------------------------
		model_time_start = timeit.default_timer()
		try:
			if args.cuda:
				img = torch.from_numpy(cv2.imread(str(p))).cuda().float()
			else:
				img = torch.from_numpy(cv2.imread(str(p))).float()
		except Exception as e:
			continue

		batch = FastBaseTransform()(img.unsqueeze(0))
		preds = net(batch)
		prog_bar.update()
		
		model_time_end = timeit.default_timer()
		model_time += (model_time_end - model_time_start)
		# ---------------------------------------------

		# Process time---------------------------------
		process_time_start = timeit.default_timer()
		h, w, _ = img.shape
		img_numpy = img.cpu().numpy().astype(np.uint8)

		t = postprocess(preds, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = args.score_threshold)

		idx = t[1].argsort(0, descending=True)[:args.top_k]
		classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]

		num_dets_to_consider = min(args.top_k, classes.shape[0])
		for j in range(num_dets_to_consider):
			if scores[j] < args.score_threshold:
				num_dets_to_consider = j
				break

		if num_dets_to_consider > 0:
			masks = masks[:num_dets_to_consider, :, :, None]

		process_time_end = timeit.default_timer()
		process_time += (process_time_end - process_time_start)
		# ---------------------------------------------

		# Length time----------------------------------
		length_time_start = timeit.default_timer()
		for i, mask in enumerate(masks):
			if classes[i] == 0:
				continue
			mask = np.squeeze(mask)
			length = mask_elli(mask)
			
			result_list.append({'Image':  str(p.name), 
								'Class':  classes[i],
								'Length': length})
			break
		else:
			result_list.append({'Image':  str(p.name), 
								'Class':  None,
								'Length': None})
		# ----------------------------------------------
		length_time_end = timeit.default_timer()
		length_time += (length_time_end - length_time_start)

	return result_list


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Fish Body Length Estimation')
	parser.add_argument('--trained_model', default='weights/yolact_fish_NewFish15_13_10000.pth', type=str)
	parser.add_argument('--config', default='yolact_NewFish15_config', type=str)
	parser.add_argument('--top_k', default=5, type=int)
	parser.add_argument('--score_threshold', default=0.3, type=float)
	parser.add_argument('--cuda', default=True, action='store_true')
	parser.add_argument('--mode', default='eval', choices = ['eval', 'infer'], type=str)
	parser.add_argument('--img_file', default='data/length_all.txt', type=str)
	parser.add_argument('--eval_csv_file', default="./results/LengthResults/yolact_new_length_ransac_kf.csv", type=str) 
	parser.add_argument('--cross_valid', default=False, action='store_true')
	parser.add_argument('--input_folder', default='data/LengthMeasure/JPEGImages', type=str)
	parser.add_argument('--output_folder', default='results/', type=str)
	parser.add_argument('--infer_csv_file', default="./results/LengthResults/yolact_new_length_infer.csv", type=str) 

	global args
	args = parser.parse_args()

	if args.config is not None:
		set_cfg(args.config)

	with torch.no_grad():
		if args.cuda:
			cudnn.fastest = True
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
		else:
			torch.set_default_tensor_type('torch.FloatTensor')

		net = Yolact()
		net.load_weights(args.trained_model)
		net.eval()

		if args.cuda:
			net = net.cuda()

		net.detect.use_fast_nms = True
		net.detect.use_cross_class_nms = False
		cfg.mask_proto_debug = False

		num_para = sum(p.numel() for p in net.parameters())
		print('Number of parameters: ', num_para)

		if args.mode == 'eval':
			with open(args.img_file, "r") as file:
				img_path = file.read().split('\n')
		elif args.mode == 'infer':
			input_folder = Path(args.input_folder)
			img_path = list(input_folder.glob('**/*'))
		else:
			exit()

		if args.mode == 'eval' and args.cross_valid:
			#random.seed(SEED)
			#random.shuffle(img_path)

			kf_eval(img_path, net)

			mae_pd = pd.DataFrame(mae_list)
			mae_mean = mae_pd.groupby('Title').mean().reset_index()
			mae_mean["KFold"] = "Average"
			mae_pd = pd.concat([mae_pd, mae_mean], axis=0, ignore_index=True)
			mae_pd.to_csv(args.eval_csv_file, header=True, columns=mae_column)
		elif args.mode == 'eval' and not args.cross_valid:
			eval(img_path, net)
		elif args.mode == 'infer':
			pred_list = infer(img_path, net)
			pred_pd = pd.DataFrame(pred_list)
			pred_pd.to_csv(args.infer_csv_file, header=True, columns=pred_column)
		else:
			exit()


	total_time = model_time + process_time + length_time
	print(f'Model time: {model_time:.3f} s')
	print(f'Process time: {process_time:.3f} s')
	print(f'Length time: {length_time:.3f} s')
	print(f'Total time: {total_time:.3f} s')
	print(f'Average time: {(total_time/len(img_path)):.3f} s')

	print('Done.')
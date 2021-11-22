import numpy as np
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import cv2
import timeit
from skimage.morphology import skeletonize, thin, medial_axis
from sklearn import linear_model
from skimage.measure import label

global image_num
image_num = 0
skel_color = np.array([0.5,0.7,0.2])
inter_color = np.array([0.22,0.7,0.6])

def multidim_intersect(arr1, arr2):
    a = set((tuple(i) for i in arr1))
    b = set((tuple(i) for i in arr2))

    return np.array(list(a.intersection(b)))

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def consecutive(data, stepsize=1):
    return len(np.split(data, np.where(np.diff(data) != stepsize)[0]+1))==1

def maskL(mask):
	global image_num
	#kernel = np.ones((17,17), np.uint8)
	#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	#dilation = cv2.dilate(closing, kernel, iterations = 5)
	#erosion = cv2.erode(dilation, kernel, iterations = 5)

	skel_mask = skeletonize((closing/255).astype(np.uint8)).astype(np.uint8)
	print(skel_mask)
	y, x = np.nonzero(mask)
	ransac = linear_model.RANSACRegressor()
	ransac.fit(x.reshape(-1,1),y)
	line_x = np.arange(0, x.max())[:,np.newaxis]
	line_y = ransac.predict(line_x)

	predict = np.column_stack((line_y.astype(np.uint8), line_x))
	predict_mask = np.zeros(mask.shape)
	for p in predict:
		predict_mask[p[0]][p[1]] = 1

	intersect = ((mask + predict_mask) > 255).astype(np.uint8)
	print(np.max(intersect))

	rgb_mask = np.stack((mask,)*3, axis = -1)
	rgb_skel = np.stack((skel_mask*255,)*3, axis = -1)
	write_mask = rgb_mask - rgb_skel + (rgb_skel * skel_color).astype(np.uint8)

	rgb_mask = np.stack((mask,)*3, axis = -1)
	rgb_inter = np.stack((intersect*255,)*3, axis = -1)
	write_mask1 = rgb_mask - rgb_inter + (rgb_inter * inter_color).astype(np.uint8)

	cv2.imwrite(f'results/newskele/closing/{image_num}.png', (closing).astype(np.uint8))
	#cv2.imwrite(f'results/newskele/dilation/{image_num}.png', (dilation).astype(np.uint8))
	#cv2.imwrite(f'results/newskele/erosion/{image_num}.png', (erosion).astype(np.uint8))
	cv2.imwrite(f'results/newskele/skel/skel{image_num}.png', write_mask)
	cv2.imwrite(f'results/newskele/newskel/skel{image_num}.png', write_mask1)
	image_num += 1



def mask_length(mask, n):
	# mask: [H x W] numpy array
	global image_num
	mask_t = mask.transpose()
	y_where = np.array([np.mean(np.argwhere(mask_row==1)) for mask_row in mask_t])
	x_where = np.isfinite(y_where)
	y_where = y_where[x_where]
	x_where = np.squeeze(np.argwhere(x_where==True))
	y_where = np.append(y_where[::n], y_where[-1])
	x_where = np.append(x_where[::n], x_where[-1])
	y_diff = y_where[1:] - y_where[:-1]
	x_diff = x_where[1:] - x_where[:-1]

	'''points = np.stack((x_where, 250 - y_where), axis=1)
	plt.imshow(mask, extent = [0,mask.shape[1],0,mask.shape[0]])
	plt.plot(points[:,0], points[:,1], linewidth=2, color='firebrick')
	plt.savefig(f'results/maskwithline/mask_line{image_num}.png')
	plt.clf()
	image_num += 1'''

	length = np.sum((y_diff**2 + x_diff**2)**0.5)
	#print(length)

	return length

def mask_length_new(mask):
	global image_num
	mask = np.array(getLargestCC(mask), dtype=np.uint8)
	#mask = cv2.dilate(mask, kernel2, iterations = 5)
	#mask = cv2.erode(mask, kernel2, iterations = 5)
	thin_mask = (thin(mask)).astype(np.uint8)

	y, x = np.nonzero(thin_mask)
	ind = np.lexsort((y,x))
	ind_part = len(ind)//6
	coord_l = np.array([[y[i], x[i]] for i in ind[:ind_part]])
	coord_r = np.array([[y[i], x[i]] for i in ind[-ind_part:]])
	
	coord_l = np.vsplit(coord_l, np.where(np.diff(coord_l[:,1]) != 0)[0]+1)
	coord_r = np.vsplit(coord_r, np.where(np.diff(coord_r[:,1]) != 0)[0]+1)
	
	l_two = False
	l_list = np.array(coord_l[0])
	
	for group in coord_l[1:]:
		cons = consecutive(group[:,0])
		
		if l_two and cons:            # split to merge
			np.vstack((l_list, group))
			break
		elif l_two and not cons:      # keep spliting
			continue
		elif not l_two and cons:      # keep merging
			continue
		elif not l_two and not cons:  # merge to split
			l_two = True
			np.vstack((l_list, group))

	r_two = False
	r_list = np.array(coord_r[0])
	
	for group in reversed(coord_r[1:]):
		cons = consecutive(group[:,0])
		
		if r_two and cons:            # split to merge
			np.vstack((r_list, group))
			break
		elif r_two and not cons:      # keep spliting
			continue
		elif not r_two and cons:      # keep merging
			continue
		elif not r_two and not cons:  # merge to split
			r_two = True
			np.vstack((r_list, group))

	left = np.mean(l_list, axis=0)
	right = np.mean(r_list, axis=0)

	length = np.linalg.norm(right-left)
	return length


def mask_morph(mask):
	# mask: [H x W] numpy array
	global image_num
	mask = np.array(getLargestCC(mask), dtype=np.uint8)
	kernel1 = np.ones((19,19), np.uint8)
	kernel2 = np.ones((5,5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
	mask = cv2.dilate(mask, kernel2, iterations = 5)
	mask = cv2.erode(mask, kernel2, iterations = 5)

	skel_mask = (skeletonize(mask) * 255).astype(np.uint8)
	thin_mask = (thin(mask) * 255).astype(np.uint8)
	medi_mask = (medial_axis(mask) * 255).astype(np.uint8)

	rgb_mask = (np.stack((mask,)*3, axis = -1)*255).astype(np.uint8)
	rgb_skel = np.stack((skel_mask,)*3, axis = -1)
	rgb_thin = np.stack((thin_mask,)*3, axis = -1)
	rgb_medi = np.stack((medi_mask,)*3, axis = -1)
	skel_mask = rgb_mask - rgb_skel + (rgb_skel * skel_color).astype(np.uint8)
	thin_mask = rgb_mask - rgb_thin + (rgb_thin * skel_color).astype(np.uint8)
	medi_mask = rgb_mask - rgb_medi + (rgb_medi * skel_color).astype(np.uint8)
	cv2.imwrite(f'results/skele/mask_with_skel{image_num}.png', skel_mask)
	cv2.imwrite(f'results/skele/mask_with_thin{image_num}.png', thin_mask)
	cv2.imwrite(f'results/skele/mask_with_medi{image_num}.png', medi_mask)
	image_num += 1

def mask_line(mask, img, w, h):
	global image_num
	contours, hierarchy = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt_index = 0
	cnt_size = 0
	for i in range(len(contours)):
		if contours[i].shape[0] > cnt_size:
			cnt_size = contours[i].shape[0]
			cnt_index = i
	cnt = contours[cnt_index]

	(vx, vy, x0, y0) = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
	constant = vx*x0 - vy*y0
	m = 1000
	#cv2.line(img, (x0-m*vx[0], y0-m*vy[0]), (x0+m*vx[0], y0+m*vy[0]), (25,180,120), 2)
	#cv2.imwrite(f'results/line/mask_with_line{image_num}.png', img)
	image_num += 1

def mask_elli(mask):
	mask = np.array(getLargestCC(mask), dtype=np.uint8)

	contours, hierarchy = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]

	elli = cv2.fitEllipseDirect(cnt)
	length = max(elli[1][0], elli[1][1])

	return length

def mask_mixed(mask, classes):
	mask = np.array(getLargestCC(mask), dtype=np.uint8)

	contours, hierarchy = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]

	# ALB, BET, BUM, DOL, LEC, MLS, OIL, SBT, SFA, SKJ, SSP, SWO, Shark, WAH, YFT
	# 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,    14,  15
	
	if classes in [11,10,13]:
		rect = cv2.fitEllipse(cnt)
		length = max(rect[1][0], rect[1][1])
	elif classes in [14]:
		rect = cv2.minAreaRect(cnt)
		length = max(rect[1][0], rect[1][1])
	else:
		elli = cv2.fitEllipseDirect(cnt)
		length = max(elli[1][0], elli[1][1])
	
	return length

def tilt_corrcet(mask):
	global image_num
	mask = np.array(getLargestCC(mask), dtype=np.uint8)
	kernel = np.ones((19,19), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	contours, hierarchy = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]

	# minAreaRect
	
	angle = cv2.minAreaRect(cnt)[-1]
	if angle < -45:
		angle = (angle+90)
	else:
		angle = angle

	dst = rotate(mask, angle)

	cv2.imwrite(f'results/rotate/mask_{image_num}.png', (mask*255).astype(np.uint8))
	cv2.imwrite(f'results/rotate/mask_rotate_{image_num}.png', (dst*255).astype(np.uint8))
	image_num += 1

def rotate(image,angle,center=None,scale=1.0):
    (w,h) = image.shape[0:2]
    if center is None:
        center = (w//2,h//2)   
    wrapMat = cv2.getRotationMatrix2D(center,angle,scale)    
    return cv2.warpAffine(image,wrapMat,(h,w))


if __name__ == '__main__':
	path = Path('data/Mask/')

	for imgpath in list(path.glob('**/*'))[:5]:
		print(imgpath)
		image = cv2.imread(str(imgpath),0)
		#maskL(image)
		#tilt_corrcet(image)
		#mask_morph(image)\
		mask_length_new(image)

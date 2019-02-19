# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 18:01:59
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:


import numpy as np

class OverlapTrack(object):

    def __init__(self, id, rect, mask, class_id,  frame_index, nb_frame_past_max):

        self.active = False;
        self.color = tuple(np.random.rand(3));
        self.id = id;
        self.last_det_index = None;
        self.mask_list = []
        self.roi_list = [];
        self.class_id = class_id;

        self._nb_frame_past_max = nb_frame_past_max;

        self.push_det(rect, mask, frame_index);


    # Push new mask
    def push_det_none(self):
        self.mask_list.insert(0, None);
        self.roi_list.insert(0, None);

        if len(self.mask_list) > self._nb_frame_past_max:
            self.mask_list.pop();

        if len(self.roi_list) > self._nb_frame_past_max:
            self.roi_list.pop();

    def push_det(self, roi, mask, frame_index):

        self.mask_list.insert(0, mask);
        self.roi_list.insert(0, roi);

        if len(self.mask_list) > self._nb_frame_past_max:
            self.mask_list.pop();

        if len(self.roi_list) > self._nb_frame_past_max:
            self.roi_list.pop();

        if not (mask is None) or not (roi is None) :
            self.last_det_index = frame_index;

    # Get Mask at index (e.g time )
    def get_mask(self, index):

        if index >= len(self.mask_list):
            return None;

        else:
            return self.mask_list[index];

    # Get Rect at index (e.g time )
    def get_roi(self, index):

        if index >= len(self.roi_list):
            return None;

        else:
            return self.roi_list[index];


########################################################
# Utilities function to compute overlap between masks
########################################################

def compute_overlap(mask_1, mask_2):
    return np.linalg.norm(np.multiply(mask_1, mask_2));

# Cost function is overlap but it should really be union minus intersection
def compute_overlap_cont(rect_1, rect_2, mask_cont_1, mask_cont_2):

    over = 0;
    if getOverlap(rect_1[[0,2]], rect_2[[0,2]]) == 0 or getOverlap(rect_1[[1,3]], rect_2[[1,3]]) == 0:
        return over;

    for i in mask_cont_1:
        for j in mask_cont_2:
            over += getOverlap(i,j);

    return over;

# def compute_union_cont(rect_1, rect_2, mask_cont_1, mask_cont_2):

#     union = 0;
#     for i in mask_cont_1:
#         for j in mask_cont_2:
#             union += getUnion(i,j);
#     return union;

# def compute_overlap_percentage_cont(rect_1, rect_2, mask_cont_1, mask_cont_2):

#     count_overlap = compute_overlap_cont(rect_1, rect_2, mask_cont_1, mask_cont_2);

#     if count_overlap == 0:
#         return 0;

#     count_union = compute_union_cont(rect_1, rect_2, mask_cont_1, mask_cont_2);
#     # return count_union;

#     percent_overlap =  float(count_overlap) / float(count_union);

#     return percent_overlap;


# def getUnion(a, b):
#     over = getOverlap(a,b);
#     return (a[1] - a[0]) + (b[1] - b[0]) - over;


def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]));


# Cost function is overlap but it should really be union minus intersection
def compute_overlap_daz(rect_1, rect_2, mask_1, mask_2):

    if getOverlap(rect_1[[0,2]], rect_2[[0,2]]) == 0 or getOverlap(rect_1[[1,3]], rect_2[[1,3]]) == 0:
        return  np.count_nonzero(mask_1) + np.count_nonzero(mask_2);


    return overlap_mask(mask_1, mask_2);

# Compute overlapping score: Slighly more complex than just overlapping
def overlap_mask(mask_1, mask_2):

    # Overlap Mask
    overlap_mask = np.logical_and(mask_1 == 1, mask_2 == 1);

    # Mask of region: Mask_i \ Overlap
    mask_count_1 = (np.logical_and(mask_1 == 1, overlap_mask == 0));
    mask_count_2 = (np.logical_and(mask_2 == 1, overlap_mask == 0));

    # Count the 1
    count_1 =  np.count_nonzero(mask_count_1);
    count_2 =  np.count_nonzero(mask_count_2);

    # Weight more the regions of mask_1 going out of the region of mask_2 (e.g we want mask_1 to be inside mask_2)
    count =  count_1 + count_2;

    return count;
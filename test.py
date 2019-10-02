import numpy as np
from tqdm import tqdm

from yolo import YOLO
from datasets_handler.pp import PP
from datasets_handler import dataset_params
from datasets_handler.voc_eval import voc_eval

bs = 16
od_thresh = 0.2

dparams = dataset_params.get_params('pp')
pp = PP(dparams, 'val', None, 'general', caching=False)

od = YOLO( bgr=True, score=od_thresh, batch_size=bs,
        model_path='model_data/pp_reanchored_best_val.h5',
        anchors_path='model_data/PP_ALL_anchors.txt',
        classes_path='model_data/PP_classes.txt' )

remainder = len(pp)
print('Forward passes...')
for start_idx in tqdm(range(0, len(pp), bs)):
    if remainder < bs:
        bs = remainder
        od.regenerate(bs)

    images, impaths = pp.pull_images(start_idx, bs)
    remainder -= len(impaths)
    results = od.detect_get_box_in(images, box_format='ltrb')

    det_dict = {}
    for i, result in enumerate(results):
        impath = impaths[i]
        for det in result:
            box_info, score, pred_cls = det
            pred_cls_id = od.class_names.index(pred_cls) + 1
            l,t,r,b = box_info
            this_array = np.array([l,t,r,b,score])
            if pred_cls_id not in det_dict:
                det_dict[pred_cls_id] = {}
            if impath not in det_dict[pred_cls_id]:
                det_dict[pred_cls_id][impath] = this_array
            else:
                det_dict[pred_cls_id][impath] = np.vstack([det_dict[pred_cls_id][impath], this_array])

# Eyes on the prizes
'''
det_dict whose key is class_id and val is cls_dict
-> cls_dict whose key is image_file_name and val is detection
    -> detection is nd array, shape (k, 5) where k = num of dets, and 5 belongs to bb infos 
        -> bb info is [xmin, ymin, xmax, ymax, conf] in pixel coords
''' 
aps = []
for cid in range(len(od.class_names)):
    aps.append( voc_eval(det_dict, pp.annotations, cid+1) )

print(aps)
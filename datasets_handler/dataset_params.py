
INCAR_PARAMS = {

    'name': 'klass_incar',

    'data_path': '/home/smiths/Sixray 500 (updated)/',

    # the dataset annotations will be saved/cached as a single file for faster loading
    # file path: cache_dir + dataset + '-' + task + '.pkl'
    # if None, caching is not used
    #'cache_dir': '../ssd-rfb/cache/',
    'cache_dir': 'cache/',
    # force reload the annotations even if it is in cache (eg due to some change)
    'reload': False,

    'img_ext': '*.jpg',

    # min bbox size, smaller bboxes will be discarded during loading
    'min_size': (10,10),
    'bag':
    {
         'classes':  ('__background__', 'plier', 'gun', 'wrench', 'scissor', 'knife'),
         'labels': {'plier': 'plier', 'gun': 'gun', 'wrench': 'wrench', 'scissor': 'scissor', 'knife': 'knife'}
     },

    # tasks, classes and labels
    'vehicle':
    {
         'classes':  ('__background__', 'vehicle'),
         'labels': {'car': 'vehicle', 'lorry': 'vehicle', 'van': 'vehicle', 'bus': 'vehicle', 'motorbike': 'vehicle', 'truck': 'vehicle', 'bicycle': 'vehicle'}
     },

    'vehicle-type':
    {
         'classes':  ('__background__', 'car', 'van', 'bus', 'lorry', 'truck', 'motorbike', 'bicycle'),
         'labels': {'car': 'car', 'van': 'van', 'bus': 'bus', 'lorry': 'lorry', 'truck': 'truck', 'motorbike': 'motorbike',  'bicycle': 'bicycle'}
     },

     'vehicle-type1':  # combines 'lorry' and 'truck' into single label 'truck'
    {
         'classes':  ('__background__', 'car', 'van', 'bus', 'truck', 'motorbike', 'bicycle'),
         'labels': {'car': 'car', 'lorry': 'truck', 'van': 'van', 'bus': 'bus', 'motorbike': 'motorbike', 'truck': 'truck', 'bicycle': 'bicycle'}
     },

    'person':
    {
         'classes':  ('__background__', 'person'),
         'labels': {'person': 'person'}
     }
}

ROSE_CCTV_PARAMS = {

    'name': 'rose_cctv',

    'data_path' : '/home/dsta/ntu-rose-cctv/',

     # the dataset annotations will be saved/cached as a single file for faster loading
    # file path: cache_dir + dataset + '-' + task + '.pkl'
    # if None, caching is not used
    'cache_dir': './cache/',
    # force reload the annotations even if it is in cache (eg due to some change)
    'reload': False,

    # min bbox size, smaller bboxes will be discarded during loading
    'min_size': (10,10),

    # tasks, classes and labels
    'vehicle':
    {
         'folder': 'Vehicle',  # the annotation folder
         'classes':  ('__background__', 'vehicle'),
         'labels': {'car': 'vehicle', 'bus': 'vehicle', 'van': 'vehicle', 'motorbike': 'vehicle', 'truck': 'vehicle', 'bike': 'vehicle', 'other': 'vehicle'}
     },

    'vehicle-type':
    {
         'folder': 'Vehicle',
         'classes':  ('__background__', 'car', 'van', 'bus', 'truck', 'motorbike', 'bicycle', 'other'),
         'labels': {'car': 'car', 'van': 'van', 'bus': 'bus', 'truck': 'truck', 'motorbike': 'motorbike', 'bike': 'bicycle', 'other': 'other'}
     },

     'vehicle-type1':  # discard the 'other' class, equivalent to vehicle-type1 task of in-car dataset
    {
         'folder': 'Vehicle',
         'classes':  ('__background__', 'car', 'van', 'bus', 'truck', 'motorbike', 'bicycle'),
         'labels': {'car': 'car', 'van': 'van', 'bus': 'bus', 'truck': 'truck', 'motorbike': 'motorbike', 'bike': 'bicycle'}
     },

    'person':
    {
         'folder': 'Person',
         'classes':  ('__background__', 'person'),
         'labels': {'person': 'person'}
     }
}

COCO_PARAMS = {

    'name': 'coco',

    'data_path' : '/media/dh/DATA4TB/Datasets/coco/',

     # the dataset annotations will be saved/cached as a single file for faster loading
    # file path: cache_dir + dataset + '-' + task + '.pkl'
    # if None, caching is not used
    'cache_dir': './cache/',
    # force reload the annotations even if it is in cache (eg due to some change)
    'reload': False,

    # min bbox size, smaller bboxes will be discarded during loading
    'min_size': (0,0),

    # tasks, classes and labels
    'general':
    {
         'classes':  ('__background__', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
         'labels': {'person': 'person', 'bicycle': 'bicycle', 'car': 'car', 'motorbike': 'motorbike', 'aeroplane': 'aeroplane', 'bus': 'bus', 'train': 'train', 'truck': 'truck', 'boat': 'boat', 'traffic light': 'traffic light', 'fire hydrant': 'fire hydrant', 'stop sign': 'stop sign', 'parking meter': 'parking meter', 'bench': 'bench', 'bird': 'bird', 'cat': 'cat', 'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep', 'cow': 'cow', 'elephant': 'elephant', 'bear': 'bear', 'zebra': 'zebra', 'giraffe': 'giraffe', 'backpack': 'backpack', 'umbrella': 'umbrella', 'handbag': 'handbag', 'tie': 'tie', 'suitcase': 'suitcase', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 'sports ball': 'sports ball', 'kite': 'kite', 'baseball bat': 'baseball bat', 'baseball glove': 'baseball glove', 'skateboard': 'skateboard', 'surfboard': 'surfboard', 'tennis racket': 'tennis racket', 'bottle': 'bottle', 'wine glass': 'wine glass', 'cup': 'cup', 'fork': 'fork', 'knife': 'knife', 'spoon': 'spoon', 'bowl': 'bowl', 'banana': 'banana', 'apple': 'apple', 'sandwich': 'sandwich', 'orange': 'orange', 'broccoli': 'broccoli', 'carrot': 'carrot', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'donut', 'cake': 'cake', 'chair': 'chair', 'sofa': 'sofa', 'pottedplant': 'pottedplant', 'bed': 'bed', 'diningtable': 'diningtable', 'toilet': 'toilet', 'tvmonitor': 'tvmonitor', 'laptop': 'laptop', 'mouse': 'mouse', 'remote': 'remote', 'keyboard': 'keyboard', 'cell phone': 'cell phone', 'microwave': 'microwave', 'oven': 'oven', 'toaster': 'toaster', 'sink': 'sink', 'refrigerator': 'refrigerator', 'book': 'book', 'clock': 'clock', 'vase': 'vase', 'scissors': 'scissors', 'teddy bear': 'teddy bear', 'hair drier': 'hair drier', 'toothbrush': 'toothbrush'}
    }
}

PP_PARAMS = {

    'name': 'pp',

    'data_path' : '/media/dh/Data/pp_modir/',

     # the dataset annotations will be saved/cached as a single file for faster loading
    # file path: cache_dir + dataset + '-' + task + '.pkl'
    # if None, caching is not used
    'cache_dir': None,
    # force reload the annotations even if it is in cache (eg due to some change)
    'reload': False,

    # min bbox size, smaller bboxes will be discarded during loading
    'min_size': (0,0),

    # tasks, classes and labels
    'general':
    {
         'classes':  ('__background__', 'ship'),
         'labels': {'ship': 'ship'}
    }
}

def get_params(dataset):
    if dataset == 'klass_incar':
        return INCAR_PARAMS
    elif dataset == 'rose_cctv':
        return ROSE_CCTV_PARAMS
    elif dataset == 'coco':
        return COCO_PARAMS
    elif dataset == 'pp':
        return PP_PARAMS
    else:
        return None

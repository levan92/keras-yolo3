import os
import cv2
import sys
import pickle

import numpy as np
import torch.utils.data as data

from PIL import Image
from tqdm import tqdm

# adapted VOC eval script to compute detection APs
# from .klass_voc_eval import voc_eval
from .voc_eval import voc_eval
def pickle_load(fname):
    with open(fname, 'rb') as pf:
         data = pickle.load(pf)
         print('Loaded {}.'.format(fname))
         return data

def pickle_save(fname, data):
    with open(fname, 'wb') as pf:
        pickle.dump(data, pf)
        print('Saved to {}.'.format(fname))

class PP(data.Dataset):

    """Loader for Microsoft's Common Objects in Context (MS-COCO) dataset

    Arguments:

    """

    def __init__(self, params, image_set, preproc=None, task='general', caching=True, run=False):
        self.params = params

        if image_set not in ['train', 'val', 'test']:
            sys.exit('Unknown image_set %s'%(image_set))

        self.image_set = image_set              # train/val/test
        self.dataset = params['name'] + '_' + image_set
        self.data_path = params['data_path']
        self.min_size = params['min_size']  # min object size, discarded if smaller

        self.preproc = preproc
        self.task = task
        self.caching = caching

        # if 'test' will only return image in get_item, no annotation
        self.phase = image_set  # train/val/test

        if task in params:
            # self.ann_folder = params[task]['folder']
            self.classes = params[task]['classes']  # e.g., params['vehicle']['classes']
            self.labels = params[task]['labels']
        else:
            sys.exit('Unknown task: %s'%(task))

        self.cls2ind = dict(zip(self.classes, range(len(self.classes))))
        self.ind2cls = {v:k for k,v in self.cls2ind.items() }

        print('\n PP DATASET:', self.dataset, self.data_path)
        print('Task:', task)
        print(self.classes)
        print(self.labels)

        if not run: 
            if self.params['reload'] or not self.load_from_cache():
                self.load_image_paths_and_annots()
            # self.image_list: full path of all images in the dataset
            # self.annotations: dict to store all annotations, keyed on full image path
            print('Number of images:', len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def num_classes(self):
        return len(self.classes)

    # def image_path(self, video_name, img_file):
    #     return self.data_path + 'images/' + video_name + '/' + img_file

    def __getitem__(self, index):
        image_path = self.image_list[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # return only the image
        if self.phase == 'test':
            if self.preproc is not None:
                img = self.preproc(img)
            return img

        # numpy array of [[xmin,ymin, xmax,ymax, class_id] ... ]
        objects = self.annotations[image_path]

        if self.preproc is not None:
            img, objects = self.preproc(img, objects)

        return img, objects


    def pull_image(self, index):
        '''Load and return image at 'index' as OpenCV image
        '''
        image_path = self.image_list[index]
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def pull_images(self, index, batch_size):
        '''Load and return image at 'index' as OpenCV image
        '''
        assert index + batch_size <= len(self.image_list), 'pull_images(): attempting to pull more images than available!'
        result = []
        impaths = []
        for i in range(index, index+batch_size):
            image_path = self.image_list[i]
            result.append( cv2.imread(image_path, cv2.IMREAD_COLOR) )
            impaths.append( image_path )
        return result, impaths
        # return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def load_image_paths_and_annots(self):
        set_file = os.path.join(self.data_path, '{}.txt'.format(self.dataset) )
        label_dir = os.path.join(self.data_path, 'labels')
        # set_file = self.data_path + 'sets/' + self.dataset + '.txt'
        with open(set_file) as fp:
            lines = fp.readlines()

        self.annotations = {}
        for line in tqdm(lines):
            image_path = line.strip()
            image_id = os.path.basename( image_path ).split('.')[0]
            label_path = os.path.join(label_dir, '{}.txt'.format(image_id))
            assert os.path.exists(image_path),'Img {} does not exist!'.format(image_path)
            assert os.path.exists(label_path),'Label file {} does not exist!'.format(label_path)
            iw, ih = Image.open(image_path).size
            with open(label_path, 'r') as f:
                label_lines = [l.strip() for l in f.readlines()]
            annot_boxes = []
            for lab in label_lines:
                classid, x_norm, y_norm, w_norm, h_norm = lab.split()
                w = float(w_norm) * iw
                h = float(h_norm) * ih
                x = float(x_norm) * iw
                y = float(y_norm) * ih

                xmin = max(0.0, x - w/2.)
                ymin = max(0.0, y - h/2.)
                xmax = min(iw-1.0, x + w/2.)
                ymax = min(ih-1.0, y +h/2.)
                if xmax - xmin < self.min_size[1]: continue
                if ymax - ymin < self.min_size[0]: continue
                annot_boxes.append([ xmin, ymin, xmax, ymax, int(classid)+1 ])
            
            if len( annot_boxes ) == 0: continue
            
            self.annotations[image_path] = np.array(annot_boxes)
        
        self.image_list = list(self.annotations.keys())

        if self.caching and self.params['cache_dir']:
            pickle_save(self.cache_file_path(), (self.image_list, self.annotations))

    def cache_file_path(self):
        if not self.params['cache_dir']: return ''

        if not os.path.exists(self.params['cache_dir']):
            os.makedirs(self.params['cache_dir'], exist_ok=True)
            print('os.makedirs:', self.params['cache_dir'])

        return self.params['cache_dir'] + self.dataset + '-' + self.task + '.pkl'

    def load_from_cache(self):
        if not self.params['cache_dir']:  # None
            return False

        cache_file = self.cache_file_path()
        if os.path.exists(cache_file):
            self.image_list, self.annotations = pickle_load(cache_file)
            print('Loaded annotations from cache', cache_file)
            print('Number of images with annotations:', len(self.image_list))
            return True

        return False

    def evaluate_detections(self, detections_dict):
        # raise NotImplementedError('TODO: we need to eval coco and not voc')
        aps = []
        print('--------------------------------')
        print('Class : \t AP (100)')
        print('--------------------------------')
        for cid, classname in enumerate(self.classes):
            if classname == '__background__': continue

            rec, prec, ap = voc_eval(detections_dict, self.annotations, cid, ovthresh=0.5)

            aps += [ap]
            print('{} : \t{:.2f}'.format(classname, 100*ap))

        mAP = 100*np.mean(aps)
        print('--------------------------------')
        print('mAP: \t{:.2f}'.format(mAP))
        print('--------------------------------')

        return aps, mAP


## test
# if __name__ == '__main__':
#
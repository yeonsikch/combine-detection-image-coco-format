import os, glob, json
import cv2
import random
import numpy as np
import copy

class CombineImageWithBbox():
    def __init__(self):
        self.prefix = '/mmdetection/data/COCO_AI_HUB/'
        self.new_prefix = '/mmdetection/data/NEW_COCO_AI_HUB_JS/'
        self.mode = 'val'
        self.save_image = False
        if not os.path.exists(self.new_prefix):
            os.mkdir(self.new_prefix)
            if not os.path.exists(self.new_prefix+'train'):
                os.mkdir(self.new_prefix+'train')
            if not os.path.exists(self.new_prefix+'val'):
                os.mkdir(self.new_prefix+'val')
            if not os.path.exists(self.new_prefix+'annotations'):
                os.mkdir(self.new_prefix+'annotations')

        self.annotations = self.prefix+f'annotations/aihub_{self.mode}_annotation.json'
        self.new_annotations = self.new_prefix+f'annotations/instances_{self.mode}.json'

        self.js = self._load_anno()

        ########test
        # self.js['images'] = self.js['images'][:20]
        # self.js['annotations'] = self.js['annotations'][:20]
        ########

        self.RESIZED_IMAGE_WIDTH = 1920
        self.RESIZED_IMAGE_HEIGHT = 1080

        self.num_of_total_image = len(self.js['images'])

        self.imageId_categoryIndex = self._make_imageId_categoryIndex()
        self.anno_id = 0
        self.new_js = copy.deepcopy(self.js)
        self.new_js['images'] = []
        self.new_js['annotations'] = []

        self._set_image_num(4)

    def _set_image_num(self, num_of_image):
        self.num_of_image = num_of_image
        self.num_of_image_by_axis = np.sqrt(self.num_of_image)

    def _load_anno(self):
        with open(self.annotations, 'r') as f:
            js = json.loads(f.read())
        return js

    def _make_imageId_categoryIndex(self):
        image_max_id = self.js['images'][-1]['id']
        imageId_categoryIndex = {x:[] for x in range(image_max_id+1)}
        print(self.js['annotations'][-1])
        print(self.js['images'][-1])
        for idx, anno in enumerate(self.js['annotations']):
            imageId_categoryIndex[anno['image_id']].append(idx)
        return imageId_categoryIndex

    def load_image(self, idx):
        image_info = self.js['images'][idx]
        file_path = self.prefix+image_info['file_path']
        this_width, this_height = image_info['width'], image_info['height']
        this_resize_width_ratio, this_resize_height_ratio = self.RESIZED_IMAGE_WIDTH/this_width, self.RESIZED_IMAGE_HEIGHT/this_height
        this_resize_width_ratio /= self.num_of_image_by_axis
        this_resize_height_ratio /= self.num_of_image_by_axis
        if self.save_image:
            img = cv2.imread(file_path)
            img = cv2.resize(img, [self.RESIZED_IMAGE_WIDTH / self.num_of_image_by_axis, self.RESIZED_IMAGE_HEIGHT / self.num_of_image_by_axis])
        else:
            img = None
        self.RESIZED_ONE_IMAGE_WIDTH = self.RESIZED_IMAGE_WIDTH / self.num_of_image_by_axis
        self.RESIZED_ONE_IMAGE_HEIGHT = self.RESIZED_IMAGE_HEIGHT / self.num_of_image_by_axis

        return image_info, img, this_resize_width_ratio, this_resize_height_ratio

    def combine_image_with_bbox(self, image_id):
        if self.num_of_image == 1:
            image_info, img, this_resize_width_ratio, this_resize_height_ratio = self.load_image(image_id)
            try:
                anno_id_list = self.imageId_categoryIndex[image_id]    
                category_id_list, bbox_list = [], []
                
                for anno_id in anno_id_list:
                    anno = self.js['annotations'][anno_id]
                    category_id = anno['category_id']
                    bbox = copy.deepcopy(anno['bbox'])
                    bbox[0], bbox[2] = bbox[0]*this_resize_width_ratio, bbox[2]*this_resize_width_ratio
                    bbox[1], bbox[3] = bbox[1]*this_resize_height_ratio, bbox[3]*this_resize_height_ratio
                    bbox = [int(x) for x in bbox]

                    category_id_list.append(category_id)
                    bbox_list.append(bbox)

                if self.save_image:
                    new_file_path = f'{self.new_prefix}/{self.mode}/'+f'combine{self.num_of_image}_'+image_info['file_name']
                    cv2.imwrite(new_file_path, img)

                image_js = {
                    'id':image_id,
                    'file_path':f'{self.mode}/combine{self.num_of_image}_'+image_info['file_name'],
                    'file_name':f'combine{self.num_of_image}_'+image_info['file_name'],
                    'height':1080,
                    'width':1920,
                }
                return image_js, category_id_list, bbox_list

            except:
                print('image__id', image_id)
                print('1 except!!!')
                pass

        indexes = [image_id] + [random.randint(0, self.num_of_total_image-1) for _ in range(self.num_of_image-1)]

        original_image_info, _, _, _ = self.load_image(image_id)

        col_tmp, row_tmp = [], []
        category_id_list, bbox_list = [], []

        for i, idx in enumerate(indexes):
            row = i//self.num_of_image_by_axis
            col = i%self.num_of_image_by_axis
            
            ## combine image
            image_info, img, this_resize_width_ratio, this_resize_height_ratio = self.load_image(idx)
            if self.save_image:
                col_tmp.append(img)
                
                if i % self.num_of_image_by_axis == self.num_of_image_by_axis-1:
                    col_tmp = cv2.hconcat(col_tmp)
                    row_tmp.append(col_tmp)
                    col_tmp = []

            ## combine bbox
            try:
                anno_id_list = self.imageId_categoryIndex[idx]
            except:
                continue

            for anno_id in anno_id_list:
                anno = self.js['annotations'][anno_id]
                category_id = anno['category_id']
                bbox = copy.deepcopy(anno['bbox'])
                # print(self.num_of_image_by_axis)
                # print('before bbox', bbox)
                bbox[0], bbox[2] = bbox[0]*this_resize_width_ratio, bbox[2]*this_resize_width_ratio
                bbox[1], bbox[3] = bbox[1]*this_resize_height_ratio, bbox[3]*this_resize_height_ratio
                # print('medium bbox', bbox)
                bbox[0] += col*self.RESIZED_ONE_IMAGE_WIDTH
                bbox[1] += row*self.RESIZED_ONE_IMAGE_HEIGHT
                bbox = [int(x) for x in bbox]
                # print('after bbox', bbox)
                # print()

                category_id_list.append(category_id)
                bbox_list.append(bbox)

        if self.save_image:
            concated_image = cv2.vconcat(row_tmp)
            new_file_path = f'{self.new_prefix}{self.mode}/'+f'combine{self.num_of_image}_'+original_image_info['file_name']
            cv2.imwrite(new_file_path, concated_image)

        image_js = {
            'id':image_id,
            'file_path':f'{self.mode}/combine{self.num_of_image}_'+original_image_info['file_name'],
            'file_name':f'combine{self.num_of_image}_'+original_image_info['file_name'],
            'height':1080,
            'width':1920,
        }

        return image_js, category_id_list, bbox_list

    def make_annotation_format(self, image_id, category_id, bbox):
        anno_js_list = []
        for i in range(len(category_id)):
            anno = {
                'id':self.anno_id,
                'image_id':image_id,
                'category_id':category_id[i],
                'bbox':bbox[i],
                'segmentation':None,
                'area':1,
                'iscrowd':0,
            }
            self.anno_id += 1
            anno_js_list.append(anno)
        return anno_js_list

    def make_json_file(self):
        num_of_image = len(self.js['images'])
        image_id_list = [x for x in range(num_of_image)]
        random.shuffle(image_id_list)
        for check_idx, image_id in enumerate(image_id_list):

            if check_idx == int(num_of_image*0.4):
                self._set_image_num(16)
            elif check_idx == int(num_of_image*0.8):
                self._set_image_num(1)
            image_js, category_id_list, bbox_list = self.combine_image_with_bbox(image_id)
            anno_js_list = self.make_annotation_format(image_id, category_id_list, bbox_list)
            self.new_js['images'].append(image_js)
            self.new_js['annotations'].extend(anno_js_list)
    
    def save_js(self):
        with open(self.new_annotations, 'w') as f:
            json.dump(self.new_js, f)

    def work(self):
        self.make_json_file()
        # print(self.js)
        self.save_js()

if __name__ == '__main__':
    random.seed(21)
    my = CombineImageWithBbox()
    my.work()
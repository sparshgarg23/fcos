# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:37:04 2022

@author: Admin
"""

from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import cv2
from tqdm import tqdm

import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms

from model.fcos import FCOSDetector

class COCOGenerator(CocoDetection):
    CLASSES_NAME = (
   '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
   'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
   'fire hydrant', 'stop sign', 'parking meter', 'bench',
   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
   'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
   'sports ball', 'kite', 'baseball bat', 'baseball glove',
   'skateboard', 'surfboard', 'tennis racket', 'bottle',
   'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
   'banana', 'apple', 'sandwich', 'orange', 'broccoli',
   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
   'couch', 'potted plant', 'bed', 'dining table', 'toilet',
   'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
   'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
   'book', 'clock', 'vase', 'scissors', 'teddy bear',
   'hair drier', 'toothbrush')
    
    def __init__(self,img_path,anno_path,resize_size=[800,1333]):
        super().__init__(img_path,anno_path)
        ids=[]
        for id in self.ids:
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)
            ann=self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids=ids
        self.category2id={v:i+1 for i,v in enumerate(self.coco.getCatIds())}
        self.id2category={v:k for k,v in self.category2id.items()}
        
        self.resize_size=resize_size
        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
    
    def __getitem__(self,idx):
        img,ann=super().__getitem__(idx)
        ann=[o for o in ann if o['iscrowd']==0]
        boxes=[o['bbox'] for o in ann]
        boxes=np.array(boxes,dtype=np.float32)
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        
        img=np.array(img)
        
        img,boxes,scale=self.preprocess_img_boxes(img,boxes,self.resize_ize)
        
        classes=[o['category_id'] for o in ann]
        classes=[self.category2id[c] for c in classes]
        
        img=transforms.ToTensor()(img)
        img=transforms.Normalize(self.mean,self.std,inplace=True)(img)
        
        classes=np.array(classes,dtype=np.int64)
        
        return img,boxes,classes,scale
    
    def preprocess_img_boxes(self,image,boxes,input_ksize):
        
        min_size,max_size=input_ksize
        h,w,_=image.shape
        
        smallest_side=min(w,h)
        largest_size=max(w,h)
        
        scale=min_size/smallest_side
        if largest_size*scale>max_size:
            scale=max_size/largest_size
        nw,nh=int(scale*w),int(scale*h)
        image_resized=cv2.resize(image,(nw,nh))
        
        pad_w=32-nw%32
        pad_h=32-nh%32
        
        image_padded=np.zeros(shape=[nh+pad_h,nw+pad_w,3],dtype=np.uint8)
        image_padded[:nh,:nw,:]=image_resized
        
        if boxes is None:
            return image_padded
        else:
            boxes[:,[0,2]]=boxes[:,[0,2]]*scale
            boxes[:,[1,3]]=boxes[:,[1,3]]*scale
            return image_padded,boxes,scale
    
    def has_only_empty_box(self,annot):
        return all(any(o<=1 for o in obj['bbox'][2:]) for obj in annot)
    
    def has_valid_annotation(self,annot):
        if len(annot)==0:
            return False
        if self.has_only_empty_box(annot):
            return False
        return True

def evaluate_coco(generator,model,thresh=0.5):
    
    results=[]
    image_idx=[]
    
    for idx in tqdm(len(generator)):
        img,gt_boxes,gt_labels,scale=generator[idx]
        scores,labels,boxes=model(img.unsqueeze(dim=0).cuda())
        scores=scores.detach().cpu().numpy()
        labels=labels.detach().cpu().numpy()
        boxes=boxes.detach().cpu().numpy()
        boxes/=scale
        boxes[:,:,2]-=boxes[:,:,0]
        boxes[:,:,3]-=boxes[:,:,1]
        
        for box,score,label in zip(boxes[0],scores[0],labels[0]):
            if score<=thresh:
                break
            image_res={
                'image_id':generator.ids[idx],
                'category_id':generator.id2category[label],
                'score':float(score),
                'bbox':box.tolist()}
            results.append(image_res)
        image_idx.append(generator.ids[idx])
    if not len(results):
        return
    json.dump(results,open('coco_bbox_results.json','w'), indent=4)
    #Load result in coco tool
    coco_true=generator.coco
    coco_pred=coco_true.loadRes('coco_bbox_results.json')
    #COCO EVAL
    coco_eval=COCOeval(coco_true,coco_pred,'bbox')
    coco_eval.params.imgIds=image_idx
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

if __name__=="__main__":
    img_path="/content/fcos/data/coco/val2017"
    anno_path="/content/fcos/data/coco/annotations/instances_val2017.json"
    
    generator=COCOGenerator(img_path, anno_path)
    
    model=FCOSDetector(mode="inference")
    model=torch.nn.DataParallel(model)
    
    model=model.cuda().eval()
    checkpoint_path=""
    model.load_state_dict(torch.load(checkpoint_path))
    
    evaluate_coco(generator,model)
    
    
            
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:01:42 2022

@author: Admin
"""

from .head import ClsCntRegHead
from .fpn_neck import FPN
from .backbone.resnet import resnet50
import torch.nn as nn
from .loss import GenTargets,LOSS,coords_fmap2orig
import torch
from .config import DefaultConfig

class FCOS(nn.Module):
    
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.backbone=resnet50(pretrained=config.pretrained,if_include_top=False)
        self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        self.head=ClsCntRegHead(config.fpn_out_channels, config.class_num,config.use_GN_head,
                                config.cnt_on_reg,config.prior)
        self.config=config
    
    def train(self,mode=True):
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname=module.__class__.__name__
            if classname.find('BatchNorm')!=-1:
                for p in module.parameters():
                    p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
    
    def forward(self,x):
        C3,C4,C5=self.backbone(x)
        all_P=self.fpn([C3,C4,C5])
        cls_logits,cnt_logits,reg_preds=self.head(all_P)
        return [cls_logits,cnt_logits,reg_preds]

class DetectionHead(nn.Module):
    
    def __init__(self,score_thresh,nms_iou_thresh,max_detection_boxes,strides,config=None):
        super().__init__()
        self.score_thresh=score_thresh
        self.nms_iou_thresh=nms_iou_thresh
        self.max_detection_boxes=max_detection_boxes
        self.strides=strides
        
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    
    def forward(self,inputs):
        '''
        given input [cls_logits,cnt_logit,reg_prediction]
        where cls_logit is batch_size*class_num*h*w
        cnt_logit is batch_size*1*h*w
        and reg_pred is batch_size*4*h*w
        '''
        
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)
        cnt_logits,=self._reshape_cat_out(inputs[1],self.strides)
        reg_preds,=self._reshape_cat_out(inputs[2],self.strides)
        
        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()
        coords=coords.cuda() if torch.cuda.is_available() else coords
        
        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)
        if self.config.add_centerness:
            cls_scores=torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))
        cls_scores=cls_classes+1
        
        boxes=self.coords2boxes(coords,reg_preds)
        
        max_num=min(self.max_detection_boxes,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])
            _boxes.append(boxes[batch][topk_ind[batch]])
        cls_scores_topk=torch.stack(_cls_scores,dim=0)
        cls_classes_topk=torch.stack(_cls_classes,dim=0)
        boxes_topk=torch.stack(_boxes,dim=0)
        
        assert boxes_topk.shape[-1]==4
        return self.post_process([cls_scores_topk,cls_classes_topk,boxes_topk])
    
    def post_process(self,preds_topk):
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_thresh
            _cls_scores_b=cls_scores_topk[batch][mask]
            _cls_class_b=cls_classes_topk[batch][mask]
            _boxes_b=boxes_topk[batch][mask]
            
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_class_b,self.nms_iou_thresh)
            
            _cls_scores_b.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_class_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes=torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        return scores,classes,boxes
    
    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        box is of dim [?,4]
        score is of dim [?,4]
        '''
        
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            
            idx=(iou<=thr).nonzero().squeeze()
            
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)
    
    def batched_nms(self,boxes,scores,idxs,iou_thresh):
        if boxes.numel()==0:
            return torch.empty((0,),dtype=torch.int64,device=boxes.device)
        
        max_coords=boxes.max()
        offsets=idxs.to(boxes)*(max_coords+1)
        boxes_for_nms=boxes+offsets[:,None]
        keep=self.box_nms(boxes_for_nms,scores,iou_thresh)
        return keep
    
    def coords2boxes(self,coords,offsets):
        '''
        coords [sum(h*w),2]
        offset [batch_size,sum(h*w),4]
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]
        boxes=torch.cat([x1y1,x2y2],dim=-1)
        return boxes
    
    def _reshape_cat_out(self,inputs,strides):
        
        '''
        inputs:list [batch_size,c,_h,_w]
        output:[batch_size,sum(h*w),c] and [sum(h*w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred, stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=-1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes
                
class FCOSDetector(nn.Module):
    
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        
        if mode=="training":
            self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range)
            self.loss_layer=LOSS()
        elif mode=="inference":
            self.detection_head=DetectionHead(config.score_threshold, config.nms_iou_threshold, config.max_detection_boxes_num, config.strides)
            self.clip_boxes=ClipBoxes()
    
    def forward(self,inputs):
        
        if self.mode=="training":
            batch_imgs,batch_boxes,batch_classes=inputs
            out=self.fcos_body(batch_imgs)
            targets=self.target_layer([out,batch_boxes,batch_classes])
            losses=self.loss_layer([out,targets])
            return losses
        elif self.mode=="inference":
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)
            scores,classes,boxes=self.detection_head(out)
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes

        
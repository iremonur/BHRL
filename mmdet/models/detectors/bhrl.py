import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_roi_extractor, build_head, HEADS
from .two_stage import TwoStageDetector
from ..plugins.match_module import MatchModule
from ..plugins.generate_ref_roi_feats import generate_ref_roi_feats
from mmcv.cnn import xavier_init
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
import cv2
from mmcv.visualization.color import color_val
from random import  choice
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin

@DETECTORS.register_module()
class BHRL(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(BHRL, self).__init__(backbone=backbone,
                                                    neck=neck,
                                                    rpn_head=rpn_head,
                                                    roi_head=roi_head,
                                                    train_cfg=train_cfg,
                                                    test_cfg=test_cfg,
                                                    pretrained=pretrained,
                                                    init_cfg=init_cfg)

        self.matching_block = MatchModule(512, 384)
    
    def matching(self, img_feat, rf_feat):
        out = []
        for i in range(len(rf_feat)):
            out.append(self.matching_block(img_feat[i], rf_feat[i]))
        return out

    def extract_feat(self, img):
        #print("start img = ", img[0].shape)
        img_feat = img[0]
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        """
        print("after backbone img = ", len(img_feat))
        print(img_feat[0].shape)
        print(img_feat[1].shape)
        print(img_feat[2].shape)
        print("last  = ", img_feat[3].shape)
        """
        rf_feat = self.backbone(rf_feat)
        """
        print("after backbone rf_img = ", len(rf_feat))
        print(rf_feat[0].shape)
        print(rf_feat[1].shape)
        print(rf_feat[2].shape)
        print("last  = ", rf_feat[3].shape)
        """
        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat = self.neck(rf_feat)
            """
            print("after neck img = ", len(img_feat))
            print(img_feat[0].shape)
            print(img_feat[1].shape)
            print(img_feat[2].shape)
            print(img_feat[3].shape)
            print("last  = ", img_feat[4].shape)
            

            print("after neck rf_img = ", len(rf_feat))
            print(rf_feat[0].shape)
            print(rf_feat[1].shape)
            print(rf_feat[2].shape)
            print(rf_feat[3].shape)
            print("last  = ", rf_feat[4].shape)
            """

        img_feat_metric = self.matching(img_feat, rf_feat) #Contrastive level?????

        #print("feat metric == ", len(img_feat_metric))
        #print(len(img_feat_metric))

        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        #print("region of interest len === ", len(ref_roi_feats))
        #print("region of interest shape === ", ref_roi_feats[0].shape)
        return tuple(img_feat_metric), tuple(img_feat), ref_roi_feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x, img_feat, ref_roi_feats = self.extract_feat(img)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_feat, ref_roi_feats, 
                                                 img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)

        return losses
        
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        #print("0 = ", img[0])
        #print("1 = ", img[1])
        #print("2 = ", img[2])
        x, img_feat, ref_roi_feats = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            #print("#################################################################")
            #print("proposal list len == ", len(proposal_list))
            #print("proposal list shape == ", proposal_list[0].shape)
            #print("#################################################################")
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)
    
    
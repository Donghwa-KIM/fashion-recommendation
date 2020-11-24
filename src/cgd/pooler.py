import torch
from detectron2.utils.events import EventStorage


class ROIpool:
    def __init__(self,detectron):
        self.detectron = detectron
        self.roi_heads_module = self.detectron.roi_heads
        
        
    def batches(self,batched_inputs):

        if self.detectron.training:
            gt_instances = [x["instances"].to(self.detectron.device) for x in batched_inputs]
        
        images = self.detectron.preprocess_image(batched_inputs)
        features, feature_lists = self.get_features(images)
        
        if self.detectron.training:
            proposals = self.get_proposals(images,features, gt_instances)     
        else:
            proposals = self.get_proposals(images,features)     

        proposals_per_batch, boxes, pred_instances = self.roi_pooling(feature_lists, proposals)

        box_feature = self.get_box_feature(feature_lists,boxes)
        if sum(proposals_per_batch)==0: 
            if 'pair_id' in batched_inputs:
                return  None, None, None, True
            else:
                return None, None, True

        # cgd target
        # pair id started from 1
        if 'pair_id' in batched_inputs:
            pair_labels = torch.LongTensor([i for data_dict ,n in zip(batched_inputs, proposals_per_batch) 
                                        for i in [data_dict['pair_id'] ]* n]).to(box_feature.device)
            return box_feature, pair_labels, pred_instances, False
        else:
            return box_feature, pred_instances, False

    
        
        
    def get_proposals(self, images, features, gt_instances=None):
        with EventStorage():
            if self.detectron.training:
                proposals, _ = self.detectron.proposal_generator(images, features, gt_instances)
                proposals = self.roi_heads_module.label_and_sample_proposals(proposals, gt_instances)
            else:
                proposals, _ = self.detectron.proposal_generator(images, features, None)
        return proposals
    
    def get_features(self, images):
        # fpn features
        features = self.detectron.backbone(images.tensor) 
        return features, [features[f] for f in self.roi_heads_module.box_in_features]
    
    def roi_pooling(self, feature_lists, proposals, is_cascade =True):
        box_features = self.roi_heads_module.box_pooler(feature_lists, [x.proposal_boxes for x in proposals])
        
        if is_cascade:
            # get last box head
            box_features = self.roi_heads_module.box_head[-1](box_features)
            predictions = self.roi_heads_module.box_predictor[-1](box_features)
            pred_instances, _ = self.roi_heads_module.box_predictor[-1].inference(predictions, proposals)

        else:
            box_features = self.roi_heads_module.box_head(box_features)
            predictions = self.roi_heads_module.box_predictor(box_features)
            pred_instances, _ = self.roi_heads_module.box_predictor.inference(predictions, proposals)
        
        proposals_per_batch = [len(ins) for ins in pred_instances]

        return proposals_per_batch, [ x.pred_boxes for x in pred_instances], pred_instances
    
    def get_box_feature(self,feature_lists, boxes):
        # cgd input
        return self.roi_heads_module.box_pooler(feature_lists, boxes)
        
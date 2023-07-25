"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict

from ops import compute_spatial_encodings, binary_focal_loss

class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_human: int, default: 15
        Number of human detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding human) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """
    def __init__(self,
        # Network components
        box_roi_pool: Module,
        box_pair_head: Module,
        box_pair_suppressor: Module,
        box_pair_predictor: Module,
        # Dataset properties
        human_idx: int,
        num_classes: int,
        # Hyperparameters
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.2,
        max_human: int = 15,
        max_object: int = 15,
        human_emotion: bool = True,
        # Misc
        distributed: bool = False
    ) -> None:
        super().__init__()

        self.box_roi_pool = box_roi_pool
        self.box_pair_head = box_pair_head
        self.box_pair_suppressor = box_pair_suppressor
        self.box_pair_predictor = box_pair_predictor
        self.human_emotion = human_emotion
        self.num_classes = num_classes
        self.human_idx = human_idx

        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.max_human = max_human
        self.max_object = max_object

        self.distributed = distributed

    def preprocess(self,
        detections: List[dict],
        targets: List[dict],
        append_gt: Optional[bool] = None,
    ) -> None:

        results = []
        test = 0
        for b_idx, detection in enumerate(detections):
            #print(f"detections are {detection}")
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']
            human_emotion = detection['human_emotion']
            #print(f"human_emotion is {human_emotion}")
            
            #print(f"human_emotions are of type {type(human_emotion)} and look like {human_emotion}")
            #print(f"boxes are {boxes}")
            #print(f"labels are {labels}")
            #print(f"scores are {scores}")
            # Append ground truth during training
            original_human_index = (labels == self.human_idx).nonzero(as_tuple=True)[0]
            if append_gt is None:
                append_gt = self.training
            if append_gt:
                target = targets[b_idx]
                n = target["boxes_h"].shape[0]
                boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
                scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
                labels = torch.cat([
                    self.human_idx * torch.ones(n, device=labels.device).long(),
                    target["object"],
                    labels
                ])
                #print(f"torch boxes shape{boxes.shape}")
                #print(f"torch scores shape {scores.shape}")
            #if self.human_emotion:
            #print(f"torch boxes are now: {boxes}")
            # Let's suppose the target['human_emotion'] is a tensor of size 3 filled with 1's
            #target_human_emotion = torch.ones(5)

            # And human_emotion is a tensor of size 3 filled with 2's
            #human_emotion = torch.full((3,), 2)
            #human_emotion = torch.cat([detection['human_emotion'],human_emotion],human_emotion)
            #print(f"at line 129, human emotions look like: {human_emotion}")
                

            # Remove low scoring examples
            active_idx = torch.nonzero(
                scores >= self.box_score_thresh
            ).squeeze(1)
            # Class-wise non-maximum suppression
            keep_idx = box_ops.batched_nms(
                boxes[active_idx],
                scores[active_idx],
                labels[active_idx],
                self.box_nms_thresh
            )
            active_idx = active_idx[keep_idx]
            # Sort detections by scores
            sorted_idx = torch.argsort(scores[active_idx], descending=True)
            active_idx = active_idx[sorted_idx]
            # Keep a fixed number of detections
            h_idx = torch.nonzero(labels[active_idx] == self.human_idx).squeeze(1)
            o_idx = torch.nonzero(labels[active_idx] != self.human_idx).squeeze(1)
            if len(h_idx) > self.max_human:
                h_idx = h_idx[:self.max_human]
            if len(o_idx) > self.max_object:
                o_idx = o_idx[:self.max_object]
            # Permute humans to the top
            keep_idx = torch.cat([h_idx, o_idx])
            active_idx = active_idx[keep_idx]
            
            active_original_human_idx = active_idx[labels[active_idx]==self.human_idx]
            active_human_idx = (original_human_index.unsqueeze(1) == active_original_human_idx).nonzero(as_tuple=True)[0]
            
            # add one dim to make even num
            human_emotion_ten= torch.cat((human_emotion, torch.tensor([0]).cuda()))
            #print(f"human_emotion is {human_emotion_ten}")
            #test = torch.zeros(8)
            #print(f"zeros tensor is shape {test.shape}")
            #print(f"human_emotion tensor is shape {human_emotion_ten.shape}")
            #exit()
            if self.human_emotion:
                results.append(dict(
                    boxes=boxes[active_idx].view(-1, 4), 
                    labels=labels[active_idx].view(-1),
                    scores=scores[active_idx].view(-1),
                    #TODO: Check if this is actually correct
                    # Do we need an index? we are appending one set of emotions to every box?
                    human_emotion= human_emotion_ten #human_emotion #human_emotion[active_human_idx],
                ))
            else:
                results.append(dict(
                    boxes=boxes[active_idx].view(-1, 4),
                    labels=labels[active_idx].view(-1),
                    scores=scores[active_idx].view(-1)
                
                ))
            #test += 1
            #if test > 5:
        #exit()
        return results

    def compute_interaction_classification_loss(self, results: List[dict]) -> Tensor:
        scores = []; labels = []
        for result in results:
            scores.append(result['scores'])
            labels.append(result['labels'])

        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            torch.cat(scores), labels, reduction='sum', gamma=0.2
        )
        return loss / n_p

    def compute_interactiveness_loss(self, results: List[dict]) -> Tensor:
        weights = []; labels = []
        for result in results:
            weights.append(result['weights'])
            labels.append(result['unary_labels'])

        weights = torch.cat(weights)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            weights, labels, reduction='sum', gamma=2.0
        )
        return loss / n_p

    def postprocess(self,
        logits_p: Tensor,
        logits_s: Tensor,
        prior: List[Tensor],
        boxes_h: List[Tensor],
        boxes_o: List[Tensor],
        object_class: List[Tensor],
        labels: List[Tensor]
    ) -> List[dict]:
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_h: List[Tensor]
            Human bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)

        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_h]

        weights = torch.sigmoid(logits_s).squeeze(1)
        scores = torch.sigmoid(logits_p)
        weights = weights.split(num_boxes)
        scores = scores.split(num_boxes)
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]

        results = []
        for w, s, p, b_h, b_o, o, l in zip(
            weights, scores, prior, boxes_h, boxes_o, object_class, labels
        ):
            # Keep valid classes
            x, y = torch.nonzero(p[0]).unbind(1)

            result_dict = dict(
                boxes_h=b_h, boxes_o=b_o,
                index=x, prediction=y,
                scores=s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach(),
                object=o, prior=p[:, x, y], weights=w
            )
            # If binary labels are provided
            if l is not None:
                result_dict['labels'] = l[x, y]
                result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)

            results.append(result_dict)

        return results

    def forward(self,
        features: OrderedDict,
        detections: List[dict],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        detections = self.preprocess(detections, targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        if self.human_emotion:
            human_emotion = [detection['human_emotion'] for detection in detections]
        box_features = self.box_roi_pool(features, box_coords, image_shapes)

        box_pair_features, boxes_h, boxes_o, object_class,\
        box_pair_labels, box_pair_prior = self.box_pair_head(
            features, image_shapes, box_features,
            box_coords, box_labels, box_scores,human_emotion, targets
        )

        box_pair_features = torch.cat(box_pair_features)
        logits_p = self.box_pair_predictor(box_pair_features)
        logits_s = self.box_pair_suppressor(box_pair_features)

        results = self.postprocess(
            logits_p, logits_s, box_pair_prior,
            boxes_h, boxes_o,
            object_class, box_pair_labels
        )

        if self.training:
            loss_dict = dict(
                hoi_loss=self.compute_interaction_classification_loss(results),
                interactiveness_loss=self.compute_interactiveness_loss(results)
            )
            results.append(loss_dict)

        return results



class MultiBranchFusion(Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `human` amd `object`
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int,
        spatial_size: int,
        representation_size: int,
        node_type: str,
        cardinality: int
    ) -> None:
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
    def _forward_object_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args) -> Tensor:
        return self._forward_method(*args)
    
    
class GraphHead(Module):
    """
    Graphical model head

    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of targe classes
    human_idx: int
        The index of human/person class in all objects
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """
    def __init__(self,
        out_channels: int,
        roi_pool_size: int,
        node_encoding_size: int, 
        representation_size: int, 
        num_cls: int, human_idx: int,
        object_class_to_target_class: List[list],
        fg_iou_thresh: float = 0.5,
        num_iter: int = 2,
    ) -> None:

        super().__init__()
        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size
        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        test = out_channels * roi_pool_size ** 2
        print(f"out_chan * roi_pool ^ 2: {test}")
        print(f"node_encoding_size: {node_encoding_size}")
        self.box_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )
        self.emotion_node_head = nn.Sequential(
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.ReLU(),
            )
        # self.emotion_head = nn.Sequential(
        #         nn.Linear(8, 8),
        #         nn.ReLU(),
        #         nn.Linear(8, 256),
        #         nn.ReLU(),
        #         #nn.Linear(node_encoding_size, node_encoding_size),
        #         nn.Linear(256, 1024),
        #         nn.ReLU(),
        #     )
        
        #self.emotion_head_proper = TestNet()
        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = MultiBranchFusion(
            256, 1024,
            representation_size, cardinality=16
        )

    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ) -> Tensor:
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self,
        x: Tensor, y: Tensor,
        scores: Tensor,
        object_class: Tensor
    ) -> Tensor:
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of human boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_h = torch.zeros(len(x), self.num_cls, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def forward(self,
        features: OrderedDict, image_shapes: List[Tuple[int, int]],
        box_features: Tensor, box_coords: List[Tensor],
        box_labels: List[Tensor], box_scores: List[Tensor],
        human_emotion: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor]
    ]:
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            box_features: Tensor
                (N, C, P, P) Pooled box features
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            box_coords: List[Tensor]
                Bounding box coordinates organised by images
            box_labels: List[Tensor]
                Bounding box object types organised by images
            box_scores: List[Tensor]
                Bounding box scores organised by images
            human_emotions: List[Tensor]
                Positional emotion scores
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_h`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]

        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_h: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        box_features = self.box_head(box_features)
        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_boxes_h = []; all_boxes_o = []; all_object_class = []
        all_labels = []; all_prior = []
        all_box_pair_features = []
        for b_idx, (coords, labels, scores,emotion) in enumerate(zip(box_coords, box_labels, box_scores,human_emotion)):
            
            n = num_boxes[b_idx]
            device = box_features.device
            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                all_box_pair_features.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                all_boxes_h.append(torch.zeros(0, 4, device=device))
                all_boxes_o.append(torch.zeros(0, 4, device=device))
                all_object_class.append(torch.zeros(0, device=device, dtype=torch.int64))
                all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter+n]
            h_node_encodings = node_encodings[:n_h]

            emotion = (emotion - 0) / (100 - 0)
            #checked here that the emotion scores were properly getting transfered

            # NOTE: Option here is to continue but may need a buffer of one extra zero read
            # in from dataset so that there is an even number for the matrix.
            
            print(f"processing {n}")
            #print(f"original emotion shape {emotion.shape}")
            #print(f"emotion is now of shape {emotion.shape}")
            #print(f"len of h_node encodings: {len(h_node_encodings)}")
            #print(f"size is {node_encodings[0].size}")

            node_emotions = self.emotion_node_head(emotion.cuda())
            print(f"node_emotions_size is {node_emotions.shape}")
            #print(f"h_node_encodings size is {h_node_encodings.shape}")
            #node_emotions_reap = node_emotions.repeat(n_h,1)
            #print(f"node_emotions_reap size: {node_emotions_reap.shape}")
            h_node_encodings = self.norm_h(h_node_encodings + node_emotions)
            #exit()
            #emotions_small = self.emotion_head(emotion.cuda())
            #print(f"emotions_small shape {emotions_small.shape}")
            #emotions_small = emotions_small.unsqueeze(0)
            #small_repeat_obj = emotions_small.repeat(n,1)
            #small_repeat_hum = emotions_small.repeat(n_h,1)
            #print(f"emotions_small shape {small_repeat_obj.shape}")
            
            
            
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )

            box_pair_spatial = self.spatial_head(box_pair_spatial)

            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_h, n, -1)

            #mod_1_emotion = emotions_small.repeat(box_pair_spatial.shape[0],1)
            #print(f"box_pair_reshaped_spatial:{box_pair_spatial_reshaped.shape}")
            #emotion_feat_proper_reshaped = emotion_feat_proper.reshape(n_h,n,-1)
            #print(f"emotion_feat_proper_reshaped: {emotion_feat_proper_reshaped.shape}")
            # model_1 = nn.Sequential(
            #     nn.Linear(8, 8),
            #     nn.ReLU(),
            #     nn.Linear(8, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, box_pair_spatial.shape[0] * 1024), #32 *1024
            #     nn.ReLU()
            # )
            #test_mod = TestNet(n_size=n)
            # print(f"initialized test_mod")
            # emotion_feat_proper = model_1(emotion)
            # emotion_feat_proper = emotion_feat_proper.view(-1, box_pair_spatial.shape[0], 1024)  # Reshape the output
            # emotion_feat_proper = torch.squeeze(emotion_feat_proper)
            # emotion_feat_proper = emotion_feat_proper.cuda()
            # print("started test mod")
            #print(f"box pair is: {tuple((box_pair_spatial_reshaped.size()))}")
            #print(f"emotion encoding is {tuple(emotion_feat.size())}")
            #test = h_node_encodings[0].shape
            #print(f"shape is {test[]}")
            #test_emotion_encoding = emotion_feat.expand(-1,h_node_encodings[0].shape)
            #print(f"emotion encoding is {tuple(test_emotion_encoding.size())}")
            adjacency_matrix = torch.ones(n_h, n, device=device)
            #emotion_feat_proper_reshaped = emotion_feat_proper.reshape(n_h, n, -1)

            for _ in range(self.num_iter):

                weights = self.attention_head(
                    torch.cat([
                        h_node_encodings[x],
                        node_encodings[y],
                    ], 1),
                    box_pair_spatial,
                    #mod_1_emotion #emotion_feat_proper
                )
                #print("got weights")
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n)

                # Update human nodes
                messages_to_h = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] *
                    self.obj_to_sub(
                        node_encodings,
                        box_pair_spatial_reshaped,
                        #small_repeat_obj #emotion_feat_proper_reshaped
                    ), dim=1)
                )
                h_node_encodings = self.norm_h(
                    h_node_encodings + messages_to_h
                )
                #print("finished one message")

                # Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_node_encodings,
                        box_pair_spatial_reshaped,
                        #small_repeat_hum #h_node_encodings #emotion_feat_proper_reshaped
                    ), dim=1)
                )
                #print("finished another message")
                node_encodings = self.norm_o(
                    node_encodings + messages_to_o
                )
            
            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )

            # One options, using a model to incorporate the features
            # model_2 = nn.Sequential(
            #     nn.Linear(8, 8),
            #     nn.ReLU(),
            #     nn.Linear(8, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, h_node_encodings[x_keep].shape[0] * 1024), #32 *1024
            #     nn.ReLU()
            # )
            # #test_mod = TestNet(n_size=n)
            # print(f"initialized test_mod")
            # emotion_feat_proper = model_2(emotion)
            # emotion_feat_proper = emotion_feat_proper.view(-1, h_node_encodings[x_keep].shape[0], 1024)  # Reshape the output
            # emotion_feat_proper = torch.squeeze(emotion_feat_proper)
            # emotion_feat_proper = emotion_feat_proper.cuda()
            # print(f"shape of encodings after message:{h_node_encodings[x_keep].shape}")
            #mod_2_emotion = emotions_small.repeat(h_node_encodings[x_keep].shape[0],1)

            all_box_pair_features.append(torch.cat([
                self.attention_head(
                #appearance features
                    torch.cat([
                        h_node_encodings[x_keep],
                        node_encodings[y_keep] #, #trying to add facial features here
                        ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep] #, mod_2_emotion #emotion_feat_proper
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])#, mod_2_emotion)
            ], dim=1))
            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            all_prior.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )

            counter += n

        return all_box_pair_features, all_boxes_h, all_boxes_o, \
            all_object_class, all_labels, all_prior

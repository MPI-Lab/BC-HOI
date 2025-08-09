import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from models.backbone import build_backbone
from models.matcher import build_matcher
from .gen import build_gen

from lavis.models import load_model_and_preprocess


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class GEN_VLKT(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_global = nn.Embedding(32, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers
        self.inter_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.obj_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.blip_ln = nn.Linear(768,256)
        self.blip_norm = nn.LayerNorm(hidden_dim)
        self.prompt2inter = nn.Linear(768, 256)
        self.inter_norm = nn.LayerNorm(256)
        self.opt_ln = nn.Linear(2560, 256)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        clip_label, obj_clip_label, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_BLIP(hoi_text_label, obj_text_label, unseen_index)
        num_obj_classes = len(obj_text) - 1  # del nothing

        self.clip_label = clip_label
        self.train_clip_label = train_clip_label
        self.obj_clip_label = obj_clip_label

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if args.with_clip_label:
            self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
            self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters()
        self.num_sp_token = 64
        self.global_token = 32
        cls_token_qformer = self.blip_model_opt.query_tokens.clone().detach().requires_grad_(False)
        self.hoi_prompt_qformer = nn.Parameter(torch.cat([cls_token_qformer[:,:1,:].repeat(1,self.global_token, 1),cls_token_qformer.repeat(1, self.num_sp_token // cls_token_qformer.shape[1], 1)],dim =1))
        # self.hoi_prompt_qformer = cls_token_qformer[:,:1,:].repeat(1, self.num_sp_token, 1)
        cls_token_clip = self.blip_model_opt.visual_encoder.cls_token.clone().detach().requires_grad_(False)
        self.hoi_prompt_clip = cls_token_clip.repeat(1, self.global_token+self.num_sp_token // cls_token_clip.shape[1], 1)
        self.embed2qformer_mask = nn.Linear(256, 256)
        self.embed2clip_mask = nn.Linear(256, 256)

        self.embed2qformer_mask_global = nn.Linear(256, 256)
        self.embed2clip_mask_global = nn.Linear(256, 256)

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_BLIP(self, hoi_text_label, obj_text_label, unseen_index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model, _, txt_processors= load_model_and_preprocess(name="blip2_feature_extractor",
                                                                              model_type="pretrain_vitL",
                                                                              is_eval=True,
                                                                              device=device)
        self.txt_processors = txt_processors["eval"]

        blip_model_opt, _,_= load_model_and_preprocess(name="blip2_opt",
                                                       model_type="pretrain_opt2.7b",
                                                       is_eval=True,
                                                       device=device)
        self.blip_model_opt = blip_model_opt

        self.seen = []
        self.unseen = []

        text_inputs = {"text_input":[self.txt_processors(hoi_text_label[id]) for id in hoi_text_label.keys()]}
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    self.unseen.append(idx)
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
                    self.seen.append(idx)
        else:
            hoi_text_label_del = hoi_text_label.copy()
            self.seen = list(range(len(hoi_text_label)))
        text_inputs_del =  {"text_input":[self.txt_processors(hoi_text_label[id]) for id in hoi_text_label_del.keys()]}

        obj_text_inputs = {"text_input":[self.txt_processors(obj_text[1]) for obj_text in obj_text_label]}
        with torch.no_grad():
            text_embedding = blip_model.extract_features(text_inputs,mode = "text",output_mode = "simple")[:,0,:]
            text_embedding_del = blip_model.extract_features(text_inputs_del,mode = "text",output_mode = "simple")[:,0,:]
            obj_text_embedding = blip_model.extract_features(obj_text_inputs,mode = "text",output_mode = "simple")[:,0,:]

        del blip_model

        return text_embedding.float(), obj_text_embedding.float(), \
               hoi_text_label_del, obj_text_inputs, text_embedding_del.float()

    def forward(self, samples: NestedTensor,targets = None, is_training=True,not_tensor_targets = None,epoch = 0):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        with torch.no_grad():
            features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape
        assert mask is not None

        blip_image_inputs ={"image":torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets],dim = 0)}
        with torch.no_grad():
            blip_image_embed = self.blip_model_opt.image_encode(blip_image_inputs,mode = 'only_qformer')
        blip_image_embed = self.blip_norm(self.blip_ln(blip_image_embed)).permute(1, 0, 2)

        h_hs, o_hs, inter_hs,global_hs,attn_map_inter,attn_map_global = self.transformer(self.input_proj(src), mask,
                                                                        self.query_embed_h.weight,
                                                                        self.query_embed_o.weight,
                                                                        self.query_embed_global.weight,
                                                                        self.pos_guided_embedd.weight,
                                                                        pos[-1],
                                                                        blip_image_embed,
                                                                        )

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()
        ### generate mask
        num_patches = 16 * 16
        num_sp_token = self.num_sp_token
        hoi_prompt_qformer = self.hoi_prompt_qformer.repeat(bs, 1, 1)
        hoi_attn_bias = attn_map_inter  # bs q 256
        hoi_attn_bias_qformer_tmp  = self.embed2qformer_mask(hoi_attn_bias)  # bs q 1+256+64
        hoi_attn_bias_clip_tmp = self.embed2clip_mask(hoi_attn_bias)# bs q 1+256+64

        hoi_attn_bias_clip_global_tmp = self.embed2clip_mask_global(attn_map_global)
        hoi_attn_bias_qformer_global_tmp = self.embed2qformer_mask_global(attn_map_global)
        global_token = self.global_token
        ### qformer mask
        hoi_attn_bias_qformer = torch.zeros([bs,global_token+num_sp_token, 1 + num_patches + global_token + num_sp_token])
        hoi_attn_bias_qformer[:,global_token:,1:1+num_patches] = hoi_attn_bias_qformer_tmp
        hoi_attn_bias_qformer[:, :global_token, 1:1 + num_patches] = hoi_attn_bias_qformer_global_tmp
        matrix_0 = (1 - torch.eye(global_token + num_sp_token, global_token + num_sp_token)) * (-10000.0)
        matrix_0 = matrix_0[None, :, :].repeat(bs, 1, 1)
        hoi_attn_bias_qformer[:, :, 1 + num_patches:] = matrix_0


        hoi_prompt_clip = self.hoi_prompt_clip.repeat(bs, 1, 1)
        clip_prompt_mask = torch.zeros([bs, 1 + num_patches + global_token +num_sp_token, 1 + num_patches + global_token + num_sp_token])
        clip_prompt_mask[:, -num_sp_token:,1:1 + num_patches] = hoi_attn_bias_clip_tmp
        clip_prompt_mask[:, :1 + num_patches, -num_sp_token-global_token:] = -10000.0
        clip_prompt_mask[:, -num_sp_token-global_token:-num_sp_token, 1:1 + num_patches] = hoi_attn_bias_clip_global_tmp
        matrix_2 = (1 - torch.eye(global_token+num_sp_token, global_token+num_sp_token)) * (-10000.0)
        matrix_2 = matrix_2[None, :, :].repeat(bs, 1, 1)
        clip_prompt_mask[:, 1 + num_patches:,1 + num_patches:] = matrix_2
        hoi_attn_bias_clip = clip_prompt_mask[:, None, :, :].repeat(1, 16, 1, 1)

        hoi_attn_bias_qformer = hoi_attn_bias_qformer.to('cuda')
        hoi_attn_bias_clip = hoi_attn_bias_clip.to('cuda')

        if is_training:
            blip_image_inputs = {"image": torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets], dim=0),
                                 "text_input": [t['llava_captions'] for t in not_tensor_targets]}
        else:
            blip_image_inputs = {"image": torch.cat([t['blip_inputs'].unsqueeze(0) for t in targets], dim=0),
                                 "text_input": ["_"] * bs}

        opt_loss, blip_prompt_embed, opt_in = self.blip_model_opt.forward_with_prompt(blip_image_inputs,
                                                                                      vis_prompt=hoi_prompt_qformer,
                                                                                      image_atts=hoi_attn_bias_qformer,
                                                                                      clip_prompt=hoi_prompt_clip,
                                                                                      clip_image_atts=hoi_attn_bias_clip,
                                                                                      io=global_token,
                                                                                      loss_mask = True)

        opt_in = self.opt_ln(opt_in).transpose(0, 1)

        warm_up = True
        if warm_up and epoch == 0:
            inter_hs = inter_hs
        else:
            inter_hs = self.inter_norm(
                inter_hs + self.prompt2inter(blip_prompt_embed)[None, :, :, :].repeat(3, 1, 1, 1))
            inter_hs = self.transformer.opt_forward(inter_hs, opt_in) + inter_hs

        train_clip_label = self.inter_adapter(self.train_clip_label) + self.train_clip_label
        clip_label = self.inter_adapter(self.clip_label) + self.clip_label
        obj_clip_label = self.obj_adapter(self.obj_clip_label) + self.obj_clip_label

        clip_label_norm = clip_label / clip_label.norm(dim=-1, keepdim=True)
        train_clip_label_norm = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
        obj_clip_label_norm = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            # outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
            outputs_obj_class = obj_logit_scale *o_hs @ obj_clip_label_norm.T
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        categories_balance = True

        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                # outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                if categories_balance:
                    outputs_hoi_class = inter_hs @ clip_label_norm.T
                    outputs_hoi_class[..., self.unseen] = outputs_hoi_class[..., self.unseen] * 0.8
                    outputs_hoi_class = logit_scale * outputs_hoi_class
                else:
                    outputs_hoi_class = logit_scale *inter_hs @ clip_label_norm.T

            else:
                # outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_hoi_class = logit_scale *inter_hs @ train_clip_label_norm.T
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        out['opt_loss'] = opt_loss

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord)


        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord):

        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1]}
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses


    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if "opt_loss" in outputs.keys():
            losses.update(outputs["opt_loss"])

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = GEN_VLKT(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['opt_loss'] = 0.1

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors

import argparse
from pathlib import Path
import numpy as np
import copy
import pickle
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
from models.blip4hoi_clip_qformer_attn_opt_loss.gen import build_gen
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
import clip

from lavis.models import load_model_and_preprocess


class GEN_VLKT(nn.Module):

    def __init__(self, backbone, transformer, num_queries, args=None):
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
        self.dec_layers = self.args.dec_layers
        self.inter_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.obj_adapter = nn.Linear(hidden_dim, hidden_dim)
        self.blip_ln = nn.Linear(768, 256)
        self.blip_norm = nn.LayerNorm(hidden_dim)
        self.prompt2inter = nn.Linear(768, 256)
        self.inter_norm = nn.LayerNorm(256)
        self.opt_ln = nn.Linear(2560, 256)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        clip_label, obj_clip_label = self.generate_vcoco_text_label()
        self.clip_label = clip_label
        self.obj_clip_label = obj_clip_label

        if args.with_clip_label:
            self.hoi_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        self.hidden_dim = hidden_dim
        self.reset_parameters()
        self.num_sp_token = 64
        self.global_token = 32
        cls_token_qformer = self.blip_model_opt.query_tokens.clone().detach().requires_grad_(False)
        self.hoi_prompt_qformer = nn.Parameter(torch.cat([cls_token_qformer[:, :1, :].repeat(1, self.global_token, 1),cls_token_qformer.repeat(1, self.num_sp_token // cls_token_qformer.shape[1], 1)],dim=1))
        # self.hoi_prompt_qformer = cls_token_qformer[:,:1,:].repeat(1, self.num_sp_token, 1)
        cls_token_clip = self.blip_model_opt.visual_encoder.cls_token.clone().detach().requires_grad_(False)
        self.hoi_prompt_clip = cls_token_clip.repeat(1,self.global_token + self.num_sp_token // cls_token_clip.shape[1],1)
        self.embed2qformer_mask = nn.Linear(256, 256)
        self.embed2clip_mask = nn.Linear(256, 256)

        self.embed2qformer_mask_global = nn.Linear(256, 256)
        self.embed2clip_mask_global = nn.Linear(256, 256)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def generate_vcoco_text_label(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        blip_model, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                  model_type="pretrain_vitL",
                                                                  is_eval=True,
                                                                  device=device)
        self.txt_processors = txt_processors["eval"]

        blip_model_opt, _, _ = load_model_and_preprocess(name="blip2_opt",
                                                         model_type="pretrain_opt2.7b",
                                                         is_eval=True,
                                                         device=device)
        self.blip_model_opt = blip_model_opt

        hoi_text_inputs = {"text_input": [self.txt_processors(vcoco_hoi_text_label[id]) for id in vcoco_hoi_text_label.keys()]}

        obj_text_inputs = {"text_input": [self.txt_processors(obj_text[1]) for obj_text in vcoco_obj_text_label]}

        with torch.no_grad():
            hoi_text_embedding = blip_model.extract_features(hoi_text_inputs,mode = "text",output_mode = "simple")[:,0,:]
            obj_text_embedding = blip_model.extract_features(obj_text_inputs,mode = "text",output_mode = "simple")[:,0,:]
        return hoi_text_embedding.float(), obj_text_embedding.float()

    def forward(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape
        assert mask is not None

        blip_image_inputs = {"image": torch.cat([(t['blip_inputs'].unsqueeze(0)).half().to("cuda") for t in targets], dim=0)}
        with torch.no_grad():
            blip_image_embed = self.blip_model_opt.image_encode(blip_image_inputs, mode='only_qformer')
        blip_image_embed = self.blip_norm(self.blip_ln(blip_image_embed)).permute(1, 0, 2)

        h_hs, o_hs, inter_hs, global_hs, attn_map_inter, attn_map_global = self.transformer(self.input_proj(src), mask,
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
        hoi_attn_bias_qformer_tmp = self.embed2qformer_mask(hoi_attn_bias)  # bs q 1+256+64
        hoi_attn_bias_clip_tmp = self.embed2clip_mask(hoi_attn_bias)  # bs q 1+256+64

        hoi_attn_bias_clip_global_tmp = self.embed2clip_mask_global(attn_map_global)
        hoi_attn_bias_qformer_global_tmp = self.embed2qformer_mask_global(attn_map_global)
        global_token = self.global_token
        ### qformer mask
        hoi_attn_bias_qformer = torch.zeros(
            [bs, global_token + num_sp_token, 1 + num_patches + global_token + num_sp_token])
        hoi_attn_bias_qformer[:, global_token:, 1:1 + num_patches] = hoi_attn_bias_qformer_tmp
        hoi_attn_bias_qformer[:, :global_token, 1:1 + num_patches] = hoi_attn_bias_qformer_global_tmp
        matrix_0 = (1 - torch.eye(global_token + num_sp_token, global_token + num_sp_token)) * (-10000.0)
        matrix_0 = matrix_0[None, :, :].repeat(bs, 1, 1)
        hoi_attn_bias_qformer[:, :, 1 + num_patches:] = matrix_0

        hoi_prompt_clip = self.hoi_prompt_clip.repeat(bs, 1, 1)
        clip_prompt_mask = torch.zeros(
            [bs, 1 + num_patches + global_token + num_sp_token, 1 + num_patches + global_token + num_sp_token])
        clip_prompt_mask[:, -num_sp_token:, 1:1 + num_patches] = hoi_attn_bias_clip_tmp
        clip_prompt_mask[:, :1 + num_patches, -num_sp_token - global_token:] = -10000.0
        clip_prompt_mask[:, -num_sp_token - global_token:-num_sp_token,
        1:1 + num_patches] = hoi_attn_bias_clip_global_tmp
        matrix_2 = (1 - torch.eye(global_token + num_sp_token, global_token + num_sp_token)) * (-10000.0)
        matrix_2 = matrix_2[None, :, :].repeat(bs, 1, 1)
        clip_prompt_mask[:, 1 + num_patches:, 1 + num_patches:] = matrix_2
        hoi_attn_bias_clip = clip_prompt_mask[:, None, :, :].repeat(1, 16, 1, 1)

        hoi_attn_bias_qformer = hoi_attn_bias_qformer.half().to('cuda')
        hoi_attn_bias_clip = hoi_attn_bias_clip.half().to('cuda')

        blip_image_inputs = {"image": torch.cat([t['blip_inputs'].unsqueeze(0).half().to("cuda") for t in targets], dim=0),
                                 "text_input": ["_"] * bs}

        opt_loss, blip_prompt_embed, opt_in = self.blip_model_opt.forward_with_prompt(blip_image_inputs,
                                                                                      vis_prompt=hoi_prompt_qformer,
                                                                                      image_atts=hoi_attn_bias_qformer,
                                                                                      clip_prompt=hoi_prompt_clip,
                                                                                      clip_image_atts=hoi_attn_bias_clip,
                                                                                      io=global_token,
                                                                                      loss_mask=True)

        opt_in = self.opt_ln(opt_in).transpose(0, 1)

        inter_hs = self.inter_norm(inter_hs + self.prompt2inter(blip_prompt_embed)[None, :, :, :].repeat(3, 1, 1, 1))
        inter_hs = self.transformer.opt_forward(inter_hs, opt_in) + inter_hs

        clip_label = self.inter_adapter(self.clip_label) + self.clip_label
        obj_clip_label = self.obj_adapter(self.obj_clip_label) + self.obj_clip_label

        clip_label_norm = clip_label / clip_label.norm(dim=-1, keepdim=True)
        obj_clip_label_norm = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            # outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
            outputs_obj_class = obj_logit_scale * o_hs @ obj_clip_label_norm.T


        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            outputs_hoi_class = logit_scale * inter_hs @ clip_label_norm.T

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}

        return out


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


class PostProcessHOI(nn.Module):

    def __init__(self, num_queries, subject_category_id, correct_mat, args):
        super().__init__()
        self.max_hois = 100

        self.num_queries = num_queries
        self.subject_category_id = subject_category_id

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta
        print('using use_nms_filter: ', self.use_nms_filter)

        self.hoi_obj_list = []
        self.verb_hoi_dict = defaultdict(list)
        self.vcoco_triplet_labels = list(vcoco_hoi_text_label.keys())
        for index, hoi_pair in enumerate(self.vcoco_triplet_labels):
            self.hoi_obj_list.append(hoi_pair[1])
            self.verb_hoi_dict[hoi_pair[0]].append(index)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_hoi_logits = outputs['pred_hoi_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
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
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in
                      zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hs = hs.to('cpu').numpy()
            os = os.to('cpu').numpy()
            os = os * os
            hs = hs + os[:, self.hoi_obj_list]
            verb_scores = np.zeros((hs.shape[0], len(self.verb_hoi_dict)))
            for i in range(hs.shape[0]):
                for k, v in self.verb_hoi_dict.items():
                    verb_scores[i][k] = np.max(hs[i, v])

            verb_labels = np.tile(np.arange(verb_scores.shape[1]), (verb_scores.shape[0], 1))

            ids = torch.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels, verb_scores)]

            current_result = {'predictions': bboxes, 'hoi_prediction': hois}

            if self.use_nms_filter:
                current_result = self.triplet_nms_filter(current_result)

            results.append(current_result)

        return results

    def triplet_nms_filter(self, preds):
        pred_bboxes = preds['predictions']
        pred_hois = preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                      str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs': [], 'objs': [], 'scores': [], 'indexes': []}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered = {
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(preds['hoi_prediction'])[all_keep_inds])
        }

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        max_scores = np.max(scores, axis=1)
        order = max_scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * HOI
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--missing_category_id', default=80, type=int)

    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)

    # * PNMS
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    parser.add_argument('--early_stop_mimic', action='store_true', help='stop mimic after step')
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--clip_model', default='/data-nas2/liaoyue/HICO-Det/ViT-B-32.pt',
                        help='clip pretrained model path')

    return parser


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    backbone = build_backbone(args)
    gen = build_gen(args)
    model = GEN_VLKT(backbone, gen, args.num_queries, args)
    post_processor = PostProcessHOI(args.num_queries, args.subject_category_id, dataset_val.correct_mat, args)
    model.to(device)
    post_processor.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=False)

    detections = generate(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)

    with open(args.save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples,targets = targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

set -x
EXP_DIR=exps/BC-HOI-HICO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        --master_port 2091 \
        main.py \
        --pretrained params/detr-r50-pre-2branch-hico.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 70 \
        --lr_drop 40 \
        --batch_size 2 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --model_name blip4hoi_clip_qformer_attn_opt_loss
#        --zero_shot_type non_rare_first/rare_first/unseen_verb/unseen_object \
#        --fix_clip \
#        --del_unseen \
#        --KO(Know_Object)
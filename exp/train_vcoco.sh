set -x
EXP_DIR=exps/BC-HOI-VCOCO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        --master_port 2091 \
        main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/vcoco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 150 \
        --lr_drop 40 \
        --batch_size 2 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --model_name blip4hoi_clip_qformer_attn_opt_loss_vcoco
#        --zero_shot_type non_rare_first/rare_first/unseen_verb/unseen_object \
#        --fix_clip \
#        --del_unseen \
#        --KO(Know_Object)
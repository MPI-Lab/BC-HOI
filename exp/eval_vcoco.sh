python generate_vcoco_official.py \
        --param_path exps/blip4hoi_clip_qformer_attn_opt_loss_vcoco/checkpoint_best.pth \
        --save_path vcoco.pickle \
        --hoi_path data/vcoco \
        --num_queries 64 \
        --dec_layers 3 \
        --use_nms_filter \
        --with_clip_label \
        --with_obj_clip_label

python vsrl_eval.py vcoco.pickle
def build_model(args):
    if args.model_name == "blip4hoi_clip_qformer_attn_opt_loss":
        from models.blip4hoi_clip_qformer_attn_opt_loss.gen_vlkt import build
    elif args.model_name == "blip4hoi_clip_qformer_attn_opt_loss_vcoco":
        from models.blip4hoi_clip_qformer_attn_opt_loss_vcoco.gen_vlkt import build
    else:
        from models.gen_vlkt import build
    return build(args)

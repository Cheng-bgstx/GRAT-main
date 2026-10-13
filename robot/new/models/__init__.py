

from .unet import T2MUnet
from .transformer import TransformerForDiffusion


__all__ = ['T2MUnet', 'TransformerForDiffusion']

def build_models(opt):
    print('\nInitializing model ...' )
    
    model_type = getattr(opt, 'model_type', 'unet')
    
    use_length_head = getattr(opt, 'end_token_exp', 'none') == 'length_head'
    max_motion_length = getattr(opt, 'max_motion_length', 490)

    # Goal adapter injection: CLI 'temb' -> model 'add', 'film' -> 'film'
    _goal_inject = getattr(opt, 'goal_adapter_injection', 'temb')
    goal_adapter_inject_type = 'add' if _goal_inject == 'temb' else _goal_inject

    # Concat 分支使用 CrossAttention（支持 cond_mask）。现有 checkpoint（exp_unet_small_concat）
    # 在训练时即用此架构，推理须保持一致；opt.txt 可能为 no_eff=False，故此处强制覆盖
    unet_goal_inject = getattr(opt, 'unet_goal_inject', 'temb')
    # NOTE: LinearCrossAttention (no_eff=False) supports cond_mask via zero-masking,
    # so concat injection does NOT require no_eff=True. Removing the forced override
    # ensures warm-start compatibility with pretrained models (which may use no_eff=False).
    # Only set no_eff=True if explicitly requested by the user.
    # if unet_goal_inject == 'concat':
    #     opt.no_eff = True

    transformer_goal_inject = getattr(opt, 'transformer_goal_inject', 'closd')

    if model_type == 'unet':
        model = T2MUnet(
            input_feats=opt.dim_pose,
            text_latent_dim=opt.text_latent_dim,
            base_dim=opt.base_dim,
            dim_mults=opt.dim_mults,
            time_dim=opt.time_dim,
            adagn=not opt.no_adagn,
            zero=True,
            no_eff=opt.no_eff,
            cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.),
            enable_goal_cond=getattr(opt, 'enable_goal_cond', False),
            unet_goal_inject=unet_goal_inject,
            goal_dim=getattr(opt, 'goal_dim', 3),
            target_joint_names=getattr(opt, 'target_joint_names', 'ROBOT_ROOT'),
            enable_length_head=use_length_head,
            max_motion_length=max_motion_length,
            use_goal_adapter=getattr(opt, 'use_goal_adapter', False),
            goal_adapter_encoder_type=getattr(opt, 'goal_adapter_encoder', 'multi'),
            goal_adapter_inject_type=goal_adapter_inject_type,
            goal_adapter_layers=getattr(opt, 'goal_adapter_layers', 1),
            text_drop_keep_goal_prob=getattr(opt, 'text_drop_keep_goal_prob', 0.0),
            use_goal_spatial_attn=getattr(opt, 'use_goal_spatial_attn', False),
        )
    elif model_type == 'unet_small':
        base_dim = getattr(opt, 'base_dim', 128)
        time_dim = getattr(opt, 'time_dim', 256)
        dim_mults = getattr(opt, 'dim_mults', [1, 2, 2, 2])
        if base_dim == 512:
            base_dim = 128
        if time_dim == 512:
            time_dim = 256
        if list(dim_mults) == [2, 2, 2, 2]:
            dim_mults = [1, 2, 2, 2]
        model = T2MUnet(
            input_feats=opt.dim_pose,
            text_latent_dim=getattr(opt, 'text_latent_dim', 256),
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            adagn=not opt.no_adagn,
            zero=True,
            no_eff=opt.no_eff,
            cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.),
            enable_goal_cond=getattr(opt, 'enable_goal_cond', False),
            unet_goal_inject=unet_goal_inject,
            goal_dim=getattr(opt, 'goal_dim', 3),
            target_joint_names=getattr(opt, 'target_joint_names', 'ROBOT_ROOT'),
            enable_length_head=use_length_head,
            max_motion_length=max_motion_length,
            use_goal_adapter=getattr(opt, 'use_goal_adapter', False),
            goal_adapter_encoder_type=getattr(opt, 'goal_adapter_encoder', 'multi'),
            goal_adapter_inject_type=goal_adapter_inject_type,
            goal_adapter_layers=getattr(opt, 'goal_adapter_layers', 1),
            text_drop_keep_goal_prob=getattr(opt, 'text_drop_keep_goal_prob', 0.0),
            use_goal_spatial_attn=getattr(opt, 'use_goal_spatial_attn', False),
        )
    elif model_type == 'transformer':
        model = TransformerForDiffusion(
            input_dim=opt.dim_pose,
            output_dim=opt.dim_pose,
            horizon=opt.max_motion_length,
            n_obs_steps=77,  # CLIP always produces 77 tokens
            clip_dim=512,
            text_latent_dim=opt.text_latent_dim,
            n_layer=getattr(opt, 'n_layer', 12),
            n_head=getattr(opt, 'n_head', 12),
            n_emb=getattr(opt, 'n_emb', 768),
            causal_attn=getattr(opt, 'causal_attn', False),
            n_cond_layers=getattr(opt, 'n_cond_layers', 0),
            cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.1),
            time_as_cond=True,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            enable_goal_cond=getattr(opt, 'enable_goal_cond', False),
            goal_dim=getattr(opt, 'goal_dim', 3),
            target_joint_names=getattr(opt, 'target_joint_names', 'ROBOT_ROOT'),
            enable_length_head=use_length_head,
            transformer_goal_inject=transformer_goal_inject,
            text_drop_keep_goal_prob=getattr(opt, 'text_drop_keep_goal_prob', 0.0),
            use_goal_spatial_attn=getattr(opt, 'use_goal_spatial_attn', False),
        )
    elif model_type == 'transformer_small':
        n_layer = getattr(opt, 'n_layer', 4)
        n_head = getattr(opt, 'n_head', 4)
        n_emb = getattr(opt, 'n_emb', 256)
        if n_layer == 12:
            n_layer = 4
        if n_head == 12:
            n_head = 4
        if n_emb == 768:
            n_emb = 256
        model = TransformerForDiffusion(
            input_dim=opt.dim_pose,
            output_dim=opt.dim_pose,
            horizon=opt.max_motion_length,
            n_obs_steps=77,
            clip_dim=512,
            text_latent_dim=getattr(opt, 'text_latent_dim', 256),
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            causal_attn=getattr(opt, 'causal_attn', False),
            n_cond_layers=getattr(opt, 'n_cond_layers', 0),
            cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.1),
            time_as_cond=True,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            enable_goal_cond=getattr(opt, 'enable_goal_cond', False),
            goal_dim=getattr(opt, 'goal_dim', 3),
            target_joint_names=getattr(opt, 'target_joint_names', 'ROBOT_ROOT'),
            enable_length_head=use_length_head,
            transformer_goal_inject=transformer_goal_inject,
            text_drop_keep_goal_prob=getattr(opt, 'text_drop_keep_goal_prob', 0.0),
            use_goal_spatial_attn=getattr(opt, 'use_goal_spatial_attn', False),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'unet', 'unet_small', 'transformer', or 'transformer_small'.")

    # Initialize stop_head for end_token_exp='stop_head'
    end_token_exp = getattr(opt, 'end_token_exp', 'none')
    if end_token_exp == 'stop_head' and hasattr(model, 'unet') and hasattr(model.unet, 'stop_head'):
        import torch.nn as nn
        # stop_head: Conv1d(feature_dim, 1, 1) operating on pre-final-conv features
        # The feature dim after last up block = dims[1] in the UNet
        unet = model.unet
        # Get the feature dimension from the final_conv's input channels
        feat_dim = unet.final_conv.in_channels
        unet.stop_head = nn.Conv1d(feat_dim, 1, 1)
        # Zero-init so stop_head has minimal impact at start
        nn.init.zeros_(unet.stop_head.weight)
        nn.init.zeros_(unet.stop_head.bias)
        print(f'[stop_head] Created Conv1d({feat_dim}, 1, 1) with zero init')

    return model



def build_pace_model(opt, pretrained_ckpt=None):
    """Build PACE model with DSSA + GR goal conditioning for 7-slot control."""
    import os as _os
    import torch as _torch

    opt.unet_goal_inject = getattr(opt, 'unet_goal_inject', 'dssa')
    opt.use_goal_spatial_attn = getattr(opt, 'use_goal_spatial_attn', True)
    opt.enable_goal_cond = getattr(opt, 'enable_goal_cond', True)
    opt.goal_adapter_injection = getattr(opt, 'goal_adapter_injection', 'temb')
    opt.target_joint_names = getattr(opt, 'target_joint_names', 'ROBOT_FULL')
    opt.end_token_exp = getattr(opt, 'end_token_exp', 'stop_head')
    opt.use_goal_adapter = getattr(opt, 'use_goal_adapter', False)

    print(f'[PACE] Model: DSSA={opt.use_goal_spatial_attn}, '
          f'inject={opt.unet_goal_inject}, target={opt.target_joint_names}')

    model = build_models(opt)

    if pretrained_ckpt and _os.path.exists(pretrained_ckpt):
        print(f'[PACE] Loading backbone from {pretrained_ckpt}')
        state = _torch.load(pretrained_ckpt, map_location='cpu')
        if 'encoder' in state:
            state = state['encoder']
        elif 'model' in state:
            state = state['model']
        elif 'model_ema' in state:
            state = state['model_ema']
        if hasattr(state, 'state_dict'):
            state = state.state_dict()

        backbone_state = {
            k: v for k, v in state.items()
            if not any(x in k for x in [
                'goal_spatial', 'embed_target_cond', 'embed_goal',
                'goal_adapter', 'goal_token', 'stop_head',
            ])
        }
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f'[PACE] Backbone: {len(backbone_state)} params loaded, '
              f'{len(missing)} new modules at init (DSSA/GR/stop_head)')

    return model

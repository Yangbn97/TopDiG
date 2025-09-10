# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import urllib.request
from functools import partial
from pathlib import Path
from models.SAM.ImageEncoder.image_encoder_baseline import ImageEncoderViT
from models.SAM.PromptEncoder import PromptEncoder
from models.sam import Sam
from models.SAM.ImageEncoder.sam_transformer import TwoWayTransformer
from models.SAM.MaskDecoder import MaskDecoder, MaskDecoderHQ
from utils.setting_utils import load_ckpt


def build_sam_vit_h(args = None, checkpoint='/data/SemanticSegmentation/checkpoints/sam_vit_h_4b8939.pth'):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )



def build_sam_vit_l(args, checkpoint='/data/SemanticSegmentation/checkpoints/sam_vit_l_0b3195.pth'):
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args, checkpoint='/root/autodl-tmp/Projects/checkpoints/sam_vit_b_01ec64.pth'):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

build_sam = build_sam_vit_b

sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def build_sam_encoder(args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    prompt_embed_dim, cls='baseline'):
    image_encoder=ImageEncoderViT(
            args= args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=args.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=args.vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            # use_rel_pos=False,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    return image_encoder

def build_sam_decoder(args,encoder_embed_dim, prompt_embed_dim, cls='baseline'):
    if cls == 'baseline':
        mask_decoder = MaskDecoder(
            num_multimask_outputs=args.class_num,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    elif cls == 'HQ':
        mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=args.class_num,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim
        )
        
    return mask_decoder

def build_sam_prompt_encoder(args, prompt_embed_dim, cls='baseline'):
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(args.image_embedding_size, args.image_embedding_size),
        input_image_size=(args.image_size, args.image_size),
        mask_in_chans=16,
    )
        
    return prompt_encoder

def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    args.vit_patch_size = vit_patch_size
    image_embedding_size = image_size // vit_patch_size
    # image_embedding_size = 1024 // 16
    args.image_embedding_size = image_embedding_size
    args.encoder_embed_dim = encoder_embed_dim
    args.prompt_embed_dim = prompt_embed_dim

    image_encoder = build_sam_encoder(args, encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, prompt_embed_dim, cls='baseline')
    mask_decoder = build_sam_decoder(args,encoder_embed_dim, prompt_embed_dim, cls='HQ')
    prompt_encoder = build_sam_prompt_encoder(args, prompt_embed_dim, cls='baseline')
    
    sam = Sam(
        args,
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[1, 1, 1],
    )
    # sam.eval(
       
    if checkpoint is not None:
        print('Loading SAM Checkpoint.....')
        sam = load_ckpt(checkpoint, sam)
        
    
    return sam

import os
import sys
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import argparse

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# Add necessary imports that were in the original code
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def build_wan22_pipeline(
    config_path: str,
    model_name: str,
    vae_path: str | None = None,
    transformer_path: str | None = None,
    transformer_high_path: str | None = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    # distributed / memory flags ↓
    device: str | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    fsdp_dit: bool = False,
    fsdp_text_encoder: bool = True,
    compile_dit: bool = False,
    GPU_memory_mode: str | None = None,
    sampler_name: str = "Flow_Unipc",
    lora_weight=1.0, lora_high_weight=1.0,
    lora_low="./Models/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
    lora_high="./Models/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
    **scheduler_extra,           # optional tweaks
):
    """
    Creates the Wan2.2 I2V pipeline *once* and returns:
    --------------------------------------------------
    pipeline    – Wan2_2I2VPipeline
    vae         – autoencoder (needed for frame-count logic)
    boundary    – float, used when calling pipeline()
    device      – torch device string
    """
    # ---------- devices ----------
    if device is None:
        device = set_multi_gpus_devices(ulysses_degree, ring_degree)

    # ---------- config ----------
    cfg = OmegaConf.load(config_path)
    boundary = cfg['transformer_additional_kwargs'].get('boundary', 0.900)

    # ---------- transformers ----------
    low_path  = os.path.join(model_name, cfg['transformer_additional_kwargs'].get(
                              'transformer_low_noise_model_subpath',  'transformer'))
    high_path = os.path.join(model_name, cfg['transformer_additional_kwargs'].get(
                              'transformer_high_noise_model_subpath', 'transformer'))

    transformer   = Wan2_2Transformer3DModel.from_pretrained(
        low_path,
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        high_path,
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # (optional) load custom checkpoints
    def maybe_load(sd_path, target):
        if sd_path is None:  return
        print(f"Loading custom checkpoint: {sd_path}")
        load_fn = torch.load if not sd_path.endswith("safetensors") else \
                  (lambda p: __import__("safetensors.torch").torch.load_file(p))
        sd = load_fn(sd_path)
        sd = sd["state_dict"] if "state_dict" in sd else sd
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print(f"  ➜ missing {len(missing)} · unexpected {len(unexpected)}")

    maybe_load(transformer_path,       transformer)
    maybe_load(transformer_high_path,  transformer_2)

    # ---------- VAE ----------
    AEClass = AutoencoderKLWan3_8 if cfg['vae_kwargs']['vae_type'] == 'Wan3_8' else AutoencoderKLWan
    vae = AEClass.from_pretrained(
        os.path.join(model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs'])
    ).to(weight_dtype)
    maybe_load(vae_path, vae)

    # ---------- tokenizer / text enc ----------
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    # ---------- scheduler ----------
    sched_map = {
        "Flow":       FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Schd = sched_map[sampler_name]
    
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        cfg['scheduler_kwargs']['shift'] = 1
    
    # convert DictConfig ➜ dict, then merge extra kwargs
    scheduler_cfg = dict(OmegaConf.to_container(cfg['scheduler_kwargs']))
    scheduler_cfg.update(scheduler_extra)          # ← safe merge
    
    scheduler = Schd(**filter_kwargs(Schd, scheduler_cfg))

    # ---------- pipeline ----------
    pipe = Wan2_2I2VPipeline(
        transformer=transformer, transformer_2=transformer_2,
        vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
    )

    # ---------- memory / dist tweaks (optional) ----------
    # preserve your original logic here …
    #   – enable multi-GPUs, FSDP, compile_dit, GPU_memory_mode, etc.

    pipe = pipe.to(device)  # final move
    
    if lora_low:
        pipe = merge_lora(pipe, lora_low,  lora_weight,      device=device)
        pipe = merge_lora(pipe, lora_high, lora_high_weight,
                          device=device, sub_transformer_name="transformer_2")
    return pipe, vae, boundary, device


def infer_video(
    pipeline,                # <- from build_wan22_pipeline()
    vae, boundary, device,   # <- from build_wan22_pipeline()
    *,
    sample_size: list[int],
    video_length: int,
    validation_image_start: str,
    prompt: str,
    save_path: str,
    negative_prompt: str = "色调艳丽，过曝，静态…",
    fps: int = 16,
    seed: int = 42,
    guidance_scale: float = 6.0,
    num_inference_steps: int = 50,
    shift: int = 5,
    # LoRA (optional) ↓
    lora_low=None, lora_high=None,
    lora_weight=1.0, lora_high_weight=1.0,
    # runtime flags ↓
    enable_riflex=False, riflex_k=6,
):
    """
    Fast per-call wrapper: *no model loading*!
    """
    generator = torch.Generator(device=device).manual_seed(seed)


    with torch.no_grad():
        # --- adjust frame count vs. VAE compression ---
        if video_length != 1:
            video_length = ((video_length - 1) //
                            vae.config.temporal_compression_ratio *
                            vae.config.temporal_compression_ratio) + 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        # riflex?
        if enable_riflex:
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

        # prepare latent + mask
        video_latent, mask_latent, _ = get_image_to_video_latent(
            validation_image_start, None,
            video_length=video_length, sample_size=sample_size
        )

        # ---- generate ----
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0], width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=video_latent, mask_video=mask_latent,
            shift=shift,
        ).videos

    # remove LoRA after generation
    # if lora_low:
    #     pipeline = unmerge_lora(pipeline, lora_low, lora_weight, device=device)
    #     pipeline = unmerge_lora(pipeline, lora_high, lora_high_weight,
    #                             device=device, sub_transformer_name="transformer_2")

    # ---- save ----
    os.makedirs(save_path, exist_ok=True)
    
    # Extract the base filename (without extension) from the input image path
    image_basename = os.path.splitext(os.path.basename(validation_image_start))[0]

    if video_length == 1:
        out_png = os.path.join(save_path, image_basename + ".png")
        img = sample[0, :, 0].permute(1, 2, 0).cpu().numpy() * 255
        Image.fromarray(img.astype(np.uint8)).save(out_png)
        return out_png
    else:
        out_mp4 = os.path.join(save_path, image_basename + ".mp4")
        save_videos_grid(sample, out_mp4, fps=fps)
        return out_mp4


def parse_args():
    parser = argparse.ArgumentParser(description='Video generation from single image and caption')
    parser.add_argument(
        '--model_name',
        type=str,
        default="./Models/Wan2.2-I2V-A14B",
        help='Path to the main model'
    )
    parser.add_argument(
        '--lora_low',
        type=str,
        default="./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
        help='Path to the low LoRA adapter'
    )
    parser.add_argument(
        '--lora_high',
        type=str,
        default="./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors",
        help='Path to the high LoRA adapter'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default="./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml",
        help='Path to the config file'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the input image'
    )
    parser.add_argument(
        '--caption',
        type=str,
        required=True,
        help='Caption/prompt for video generation'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default="./output/ffgo_eval",
        help='Path to save output videos'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Video height in pixels (default: 480)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Video width in pixels (default: 640)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default=None,
        help='Video resolution as HEIGHTxWIDTH (e.g., 720x1280). Overrides --height and --width if provided.'
    )
    parser.add_argument(
        '--video_length',
        type=int,
        default=81,
        help='Number of frames in the output video (default: 81)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for generation (default: 6.0)'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps (default: 50)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Verify image path exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    # Parse resolution
    if args.resolution:
        try:
            height, width = map(int, args.resolution.split('x'))
        except ValueError:
            raise ValueError(f"Invalid resolution format: {args.resolution}. Use HEIGHTxWIDTH (e.g., 720x1280)")
    else:
        height = args.height
        width = args.width
    
    sample_size = [height, width]
    print(f"Using resolution: {width}x{height}")
    
    # Negative prompt
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    # Add prefix to caption
    prompt = "ad23r2 the camera view suddenly changes. " + args.caption
    
    print(f"Input image: {args.image_path}")
    print(f"Caption: {args.caption}")
    print(f"Full prompt: {prompt}")
    
    # Build pipeline once
    pipe, vae, boundary, device = build_wan22_pipeline(
        config_path=args.config_path,
        model_name=args.model_name,
        lora_low=args.lora_low,
        lora_high=args.lora_high,
    )
    
    # Generate video
    video_path = infer_video(
        pipe, vae, boundary, device,
        sample_size=sample_size,
        video_length=args.video_length,
        validation_image_start=args.image_path,
        prompt=prompt,
        save_path=args.output_path,
        negative_prompt=negative_prompt,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )
    print(f"Video saved to: {video_path}")
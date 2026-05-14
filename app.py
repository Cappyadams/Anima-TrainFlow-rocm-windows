import os
import re
import json
import platform
import subprocess
import toml
import glob
import math
from pathlib import Path
import sys
import gradio as gr
import psutil
from PIL import Image

CSS = """
.gradio-container {
    position: relative !important;
}

#main-header {
    margin-bottom: -10px !important;
}

.attribution {
    position: absolute;
    right: 25px;
    top: 5px;
    z-index: 99999;
    pointer-events: none;
}

.author-text {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(187, 154, 247, 0.6);
    letter-spacing: 0.5px;
}

#log-container textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
    height: 385px !important;
    border: 1px solid #2f334d !important;
}

*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
}
*::-webkit-scrollbar-thumb {
    background: #7aa2f7 !important;
    border-radius: 10px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: #bb9af7 !important;
}

footer {
    display: none !important;
}

"""

JS_SCROLL = """
function() {
    setTimeout(() => {
        const el = document.querySelector('#log-container textarea');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, 50);
}
"""


LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


LOG_BOX__MAX_LINES = 16
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500


ROOT = Path(__file__).resolve().parent
PORTABLE_PYTHON = ROOT / "python_embeded" / "python.exe"

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts" 
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"

training_process = None

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = {
    "trigger_word": "",
    "dataset_path": "",
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-preview.safetensors"),
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 16,
    "learning_rate": "1.0", # Prodigy default
    "optimizer": "Prodigy", # Prodigy default
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
}
def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def auto_save_state(*args):
    keys = list(DEFAULT_SETTINGS.keys())
    current_state = dict(zip(keys, args))
    save_settings(current_state)


HIDDEN_SETTINGS = {
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "train_batch_size": 1,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "timestep_sampling": "logit_normal",
    "discrete_flow_shift": 3.0,
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs_to_disk": True,
    "cache_text_encoder_outputs": True,
    "sdpa": True,
    "weighting_scheme": "logit_normal",
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": True,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "blocks_to_swap": 0,
}


def analyze_dataset_resolution(dataset_path):
    default_base = 512
    default_max_bucket = 1024

    if not dataset_path or not os.path.exists(dataset_path):
        return default_base, default_max_bucket

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in Path(dataset_path).rglob('*') if f.suffix.lower() in valid_exts]

    if not image_files:
        return default_base, default_max_bucket

    areas = []
    max_side = 0
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                areas.append(w * h)
                max_side = max(max_side, w, h)
        except Exception: pass 

    if not areas: return default_base, default_max_bucket

    areas.sort()
    median_area = areas[len(areas) // 2]
    
    equivalent_side = math.sqrt(median_area)
    base_res = int(round(equivalent_side / 64.0) * 64)
    base_res = max(512, min(1024, base_res))

    max_side_rounded = int(math.ceil(max_side / 64.0) * 64)
    max_bucket = max(base_res + 256, max_side_rounded)
    max_bucket = min(1536, max_bucket) 

    return base_res, max_bucket


def create_sample_prompts(project_name, trigger_word, pos_prompt, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()
    user_prompt = pos_prompt.strip().replace("\n", " ")
    
    if trigger and not user_prompt.startswith(trigger):
        actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
    else:
        actual_pos = user_prompt if user_prompt else trigger

    actual_neg = neg_prompt.strip().replace("\n", " ")
    prompt_str = f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
    with open(prompt_path, "w", encoding="utf-8") as f: f.write(prompt_str)
    return str(prompt_path)

def create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, out_dir):
    config_path = out_dir / f"{project_name}_dataset.toml"
    prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else None
    dataset_config = {
        "general": {"enable_bucket": True, "min_bucket_reso": 256, "max_bucket_reso": max_bucket, "bucket_reso_steps": 64, "bucket_no_upscale": False},
        "datasets": [{
            "resolution": base_res, 
            "subsets": [{"image_dir": Path(dataset_path).resolve().as_posix(), "caption_extension": ".txt", "num_repeats": 1000, "caption_prefix": prefix, "keep_tokens": 1, "caption_dropout_rate": 0.05}]
        }]
    }
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(dataset_config, f)
    return str(config_path)

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed):
    config_path = config_save_dir / f"{project_name}_training.toml"
    network_alpha = max(1, int(rank) // 2)
    
    opt_args = ["weight_decay=0.01"]
    if optimizer == "Prodigy":
        scheduler = "constant"
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        scheduler = "cosine"

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_dim": int(rank),
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "max_train_steps": int(max_steps),
        "train_batch_size": HIDDEN_SETTINGS["train_batch_size"],
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
        "sample_every_n_steps": int(sample_steps),
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(training_config, f)
    return str(config_path)


def get_latest_images(sample_dir):
    if not sample_dir.exists(): return []
    images = glob.glob(str(sample_dir / "*.png")) + glob.glob(str(sample_dir / "*.jpg")) + glob.glob(str(sample_dir / "*.webp"))
    images.sort(key=os.path.getmtime, reverse=True)
    
    return [(img, Path(img).name) for img in images]

def start_training(trigger_word, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, t_steps, save_steps, sample_steps, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed):
    global training_process

     # --- PATH VALIDATION BLOCK ---
    error_messages = []
    
    # Check DiT file
    if not dit_p or not os.path.isfile(dit_p):
        error_messages.append(f"❌ DiT file not found: {dit_p}")
    
    # Check Qwen file
    if not qwen_p or not os.path.isfile(qwen_p):
        error_messages.append(f"❌ Qwen3 file not found: {qwen_p}")
        
    # Check VAE file
    if not vae_p or not os.path.isfile(vae_p):
        error_messages.append(f"❌ VAE file not found: {vae_p}")

    # Check Dataset directory
    if not dataset_path or not os.path.exists(dataset_path):
        error_messages.append(f"❌ Dataset path not found: {dataset_path}")

    if error_messages:
        full_error = "\n".join(error_messages)
        full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models <- Set Once'"
        yield full_error, gr.update()
        return
    # --- END VALIDATION BLOCK ---
    
    if training_process is not None and training_process.poll() is None:
        yield "⚠️ Training is already running!", gr.update()
        return

    project_name = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_') or "untitled"

    project_out_dir = OUTPUT_BASE / project_name
    sample_dir = project_out_dir / "sample"
    project_configs_dir = project_out_dir / "configs"

    for d in [project_out_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_lines = [f"🚀 Preparing: {project_name}..."]
    last_image_count = 0
    step_pattern = re.compile(r"(\d+)/(\d+)") 
    
    yield "\n".join(log_lines), gr.update()

    log_lines.append(f"🔍 Analyzing dataset images...")
    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    log_lines.append(f"📐 Auto-Resolution Set: Base {base_res}px, Max Bucket {max_bucket}px")

    models = {"dit_path": dit_p, "qwen_path": qwen_p, "vae_path": vae_p}
    prompt_path = create_sample_prompts(project_name, trigger_word, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, project_configs_dir)
    dataset_toml = create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, project_configs_dir)
    training_toml = create_training_toml(project_name, project_configs_dir, project_out_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed)

    cmd = [
        str(PORTABLE_PYTHON.resolve()), "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        TRAIN_SCRIPT.resolve().as_posix(), 
        "--config_file", Path(training_toml).resolve().as_posix(), 
        "--dataset_config", Path(dataset_toml).resolve().as_posix()
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"

    try:
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, cwd=str(TRAIN_DIR.resolve()), env=env, encoding="utf-8", errors="ignore")
        
        
        for line in iter(training_process.stdout.readline, ""):
            line_str = line.replace('\r', '').strip()
            if not line_str: continue

            
            if any(skip_word in line_str for skip_word in LOG_BLACKLIST):
                continue 

            
            if "subprocess.CalledProcessError" in line_str and "returned non-zero exit status 15" in line_str:
                log_lines.append("🛑 Interrupted by user.")
                yield "\n".join(log_lines), gr.update()
                break 

           
            if "steps:" in line_str and "/" in line_str:
                match = step_pattern.search(line_str)
                if match:
                    current_step_info = match.group(0)
                    if log_lines and "steps:" in log_lines[-1] and current_step_info in log_lines[-1]:
                        log_lines[-1] = line_str
                    else:
                        log_lines.append(line_str)
                else:
                    log_lines.append(line_str)
            else:
                log_lines.append(line_str)
            
            if len(log_lines) > MAX_LOG_LINES: del log_lines[:-MAX_LOG_LINES]


            check_image = any(x in line_str.lower() for x in ["saved", "sample", "%|", "it/s", "s/it"])
            if check_image:
                current_images = get_latest_images(sample_dir)
                if len(current_images) != last_image_count:
                    last_image_count = len(current_images)
                    yield "\n".join(log_lines), current_images
                    continue

            yield "\n".join(log_lines), gr.update()
            
        if training_process is not None:
            training_process.wait()
            
        log_lines.append("✅ Process finished or stopped.")
        yield "\n".join(log_lines), get_latest_images(sample_dir)
        
    except Exception as e:
        log_lines.append(f"❌ Error: {str(e)}")
        yield "\n".join(log_lines), gr.update()
    finally:
        training_process = None

def stop_training():
    global training_process
    if training_process is not None:
        try:
            parent = psutil.Process(training_process.pid)
            for child in parent.children(recursive=True):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=3)
            for survival in alive:
                try: survival.kill()
                except psutil.NoSuchProcess: pass
            parent.terminate()
            parent.wait(timeout=3)
            return "🛑 Stopping training... (Clearing VRAM)"
        except psutil.NoSuchProcess:
            return "ℹ️ Process already finished."
        except Exception as e:
            return f"⚠️ Error during stop: {str(e)}"
    return "ℹ️ Not running."

def open_output_folder(trigger_word):
    proj = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_')
    target_dir = OUTPUT_BASE / proj if proj else OUTPUT_BASE
    if not target_dir.exists(): target_dir = OUTPUT_BASE
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    return "📁 Folder opened."

def handle_optimizer_change(opt, current_lr, saved_adam_lr):
    if opt == "Prodigy": return "1.0", current_lr
    return (saved_adam_lr if current_lr == "1.0" else current_lr), saved_adam_lr


cs = load_settings()

with gr.Blocks(title="Anima TrainFlow: Easy LoRA Trainer for Anima 2B") as ui:
    gr.Markdown(
        "# Anima TrainFlow\n"
        '<div class="attribution"><span class="author-text">Created by ThetaCursed</span></div>',
        elem_id="main-header"
    )
    
    saved_adam_lr = gr.State(value="0.00005")

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    trigger_word = gr.Textbox(label="Trigger Word / Project Name", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style")
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset")
                with gr.Accordion("🔧 Paths to Models <- Set Once", open=False):
                    dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""))
                    qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""))
                    vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""))
                with gr.Row():
                    start_btn = gr.Button("🚀 Start", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                    folder_btn = gr.Button("📁 Checkpoint Folder", variant="secondary")
            with gr.Column(scale=1):
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0)
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", "1.0"))
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=["Prodigy", "AdamW8bit", "AdamW"], value=cs.get("optimizer", "Prodigy"))
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                with gr.Row():
                    steps_input = gr.Number(label="Training Steps", value=cs.get("training_steps", 2400), precision=0)
                    save_steps_input = gr.Number(label="Save Every n Steps", value=cs.get("save_steps", 300), precision=0)
                    sample_steps_input = gr.Number(label="Preview Every n Steps", value=cs.get("sample_steps", 300), precision=0)

    with gr.Row():
        with gr.Column(scale=1):
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
        with gr.Column(scale=1):
            preview_gallery = gr.Gallery(label="Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain")
            with gr.Group():
                pos_prompt = gr.Textbox(label="Prompt (Trigger word added automatically)", lines=2, value=cs.get("pos_prompt", ""))
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""))
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80)
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80)
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80)
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80)
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80)

    inputs_list = [
        trigger_word, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input, 
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val
    ]

    def load_state_on_refresh():
        current_settings = load_settings()
        return [current_settings.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS.keys()]

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=inputs_list)    

    output_log.change(None, None, None, js=JS_SCROLL)
    optimizer_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, lr_input, saved_adam_lr], outputs=[lr_input, saved_adam_lr])
    for comp in inputs_list: comp.change(fn=auto_save_state, inputs=inputs_list)

    start_btn.click(fn=start_training, inputs=inputs_list, outputs=[output_log, preview_gallery])
    stop_btn.click(fn=stop_training, outputs=output_log)
    folder_btn.click(fn=open_output_folder, inputs=[trigger_word], outputs=output_log)

if __name__ == "__main__":
     ui.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
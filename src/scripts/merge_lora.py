import os
import json
import shutil
import torch
import argparse
from peft import PeftModel, PeftConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from safetensors.torch import load_file as safe_load_file

def apply_lora(model_name_or_path, lora_path, output_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = BASE_TYPE.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map="auto" 
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    print(f"Loading the LoRA configuration from {lora_path}")
    peft_config = PeftConfig.from_pretrained(lora_path)
    peft_config.inference_mode = True
    
    lora_model = PeftModel(base, peft_config)

    print("Fixing and loading LoRA weights...")
    adapter_path_bin = os.path.join(lora_path, "adapter_model.bin")
    adapter_path_safe = os.path.join(lora_path, "adapter_model.safetensors")
    
    adapters_weights = {}
    if os.path.exists(adapter_path_safe):
        adapters_weights = safe_load_file(adapter_path_safe)
    elif os.path.exists(adapter_path_bin):
        adapters_weights = torch.load(adapter_path_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"Cannot find adapter_model.bin or adapter_model.safetensors in {lora_path}")

    corrected_weights = {}
    for k, v in adapters_weights.items():
        if "base_model.model." in k:
            print("success")
            new_k = k.replace("encoder.", "")
            corrected_weights[new_k] = v
        else:
            corrected_weights[k] = v
    save_path = os.path.join(lora_path, "corrected_weights.pt")
    torch.save(corrected_weights, save_path)

    set_peft_model_state_dict(lora_model, corrected_weights)

    print("Merging the LoRA...")
    # 6. 合并并卸载
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora-path", type=str, default="./outputs/laser-qwen3-0.6b/checkpoint-1276")
    output_path = args.lora_path + "-merged"
    args = parser.parse_args()  
    BASE_TYPE = AutoModelForCausalLM 
    apply_lora(args.base_model_path, args.lora_path, output_path)
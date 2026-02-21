base_model_name="Qwen3-0.6B"
base_model_path="Qwen/Qwen3-0.6B"
thinking_steps=3
bs=8
gd_steps=2
data_path='./data/bge_reasoning'
output_path='./outputs/laser-qwen3-0.6b'

# assume in the root path
PYTHONPATH=src deepspeed --module tevatron.retriever.driver.train_laser \
  --deepspeed /src/scripts/ds_stage0.json \
  --output_dir ${output_path} \
  --model_name_or_path ${base_model_path} \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --dataset_name ${data_path} \
  --query_prefix "Instruct: {instruct}\nQuery: {query}" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --padding_side left \
  --normalize \
  --add_chat_template \
  --temperature 0.02 \
  --per_device_train_batch_size ${bs} \
  --train_group_size 2 \
  --learning_rate 1e-4 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --lora_r 64 \
  --lora_alpha 32 \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${gd_steps} \
  --num_thinking_steps ${thinking_steps} \
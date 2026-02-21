export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

model_path="./outputs/laser-qwen3-0.6b/checkpoint-1276-merged"
model_name="laser-qwen3-0.6b"
num_thinking_steps=3


python eval/run_mteb.py \
  --model ${model_path} \
  --model_type "laser" \
  --num_thinking_steps ${num_thinking_steps} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"add_eos_id\": false, \"use_instruction\": true, \"instruction_template\": \"Instruct: {task_description}\nQuery: {query}\", \"instruction_dict_path\": \"task_prompts_multilingual.json\", \"trust_remote_code\": true, \"attn_implementation\":\"eager\", \"add_chat_template\": true}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir /mnt/workspace/jiajie/reasoning/results/${model_name} \
  --benchmark "BRIGHT" $@ \
  --batch_size 8

python eval/run_mteb.py \
  --model ${model_path} \
  --model_type "laser" \
  --num_thinking_steps ${num_thinking_steps} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"add_eos_id\": false, \"use_instruction\": true, \"instruction_template\": \"Instruct: {task_description}\nQuery: {query}\", \"instruction_dict_path\": \"task_prompts_multilingual.json\", \"trust_remote_code\": true, \"attn_implementation\":\"eager\", \"add_chat_template\": true}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir /mnt/workspace/jiajie/reasoning/results/${model_name} \
  --benchmark "FollowIR" $@ \
  --batch_size 8

python eval/run_mteb.py \
  --model ${model_path} \
  --model_type "laser" \
  --num_thinking_steps ${num_thinking_steps} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"add_eos_id\": false, \"use_instruction\": true, \"instruction_template\": \"Instruct: {task_description}\nQuery: {query}\", \"instruction_dict_path\": \"task_prompts_multilingual.json\", \"trust_remote_code\": true, \"attn_implementation\":\"eager\", \"add_chat_template\": true}" \
  --run_kwargs "{\"save_predictions\": \"true\"}" \
  --output_dir /mnt/workspace/jiajie/reasoning/results/${model_name} \
  --benchmark "BrowseCompPlus" $@ \
  --batch_size 8
exit

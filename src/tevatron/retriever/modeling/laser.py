import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from .encoder import EncoderModel, EncoderOutput

logger = logging.getLogger(__name__)


class LaSERModel(EncoderModel):
    """
    LaSERModel: A retriever model leveraging Large Language Models with iterative 
    thinking steps and knowledge distillation for enhanced representation learning.
    """
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(
        self,
        encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        pooling: str = 'cls',
        normalize: bool = False,
        temperature: float = 0.02,
        num_thinking_steps: int = 3,
        kl_weight: float = 10.0,
        latent_weight: float = 0.1,
        latent_temperature: float = 0.05,
    ):
        """
        Args:
            encoder: The backbone language model (e.g., Llama, Qwen).
            tokenizer: The tokenizer corresponding to the encoder.
            pooling: Pooling strategy for representations.
            normalize: Whether to L2 normalize the output representations.
            temperature: Temperature for contrastive loss.
            num_thinking_steps: Number of reasoning/thinking steps to simulate.
            kl_weight: Weight for the KL divergence loss against teacher scores.
            latent_weight: Weight for the latent step alignment loss.
            latent_temperature: Temperature for the latent alignment KL loss.
        """
        super(LaSERModel, self).__init__(encoder, pooling, normalize, temperature)
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.temperature = temperature
        self.eos_token_id = tokenizer.eos_token_id
        self.max_num_thinking_steps = num_thinking_steps
        
        # Hyperparameters for loss calculation (Removed magic numbers)
        self.kl_weight = kl_weight
        self.latent_weight = latent_weight
        self.latent_temperature = latent_temperature

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def encode_teacher(
        self, 
        qry: Dict[str, torch.Tensor], 
        original_query_attention_mask: torch.Tensor, 
        num_thinking_steps: int, 
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the query using the teacher model and extracts aligned reasoning features."""
        batch_size = qry['input_ids'].shape[0]

        query_embeddings = self.encoder.get_input_embeddings()(qry['input_ids'])
        embedding_table = self.encoder.get_input_embeddings().weight

        eos_embedding = embedding_table[self.eos_token_id].view(1, 1, -1).expand(batch_size, -1, -1)
        query_embeddings = torch.cat([query_embeddings, eos_embedding], dim=1)
        
        mask_padding = torch.ones(batch_size, 1, dtype=torch.long, device=qry['input_ids'].device)
        full_attention_mask = torch.cat([qry['attention_mask'], mask_padding], dim=1)
        
        outputs = self.encoder(
            inputs_embeds=query_embeddings,
            attention_mask=full_attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )

        reps = outputs.hidden_states[-1][:, -1, :]
        reps = F.normalize(reps, p=2, dim=-1)

        original_query_length = original_query_attention_mask.sum(dim=1).long()
        teacher_query_length = qry['attention_mask'].sum(dim=1).long()
        teacher_mask_num = qry['attention_mask'].size(1) - teacher_query_length
        start_idx_list = original_query_length + teacher_mask_num

        aligned_features = []
        newline_ids = self.tokenizer.encode('\n\n', add_special_tokens=False)
        newline_id = newline_ids[-1] if newline_ids else self.tokenizer.eos_token_id

        for b in range(batch_size):
            start_idx = start_idx_list[b]
            cot_ids = qry['input_ids'][b, start_idx:]
            cot_reps = outputs.hidden_states[-1][b, start_idx:, :]
            
            if cot_ids.size(0) == 0:
                feature = outputs.hidden_states[-1][b, -1, :].unsqueeze(0).expand(num_thinking_steps, -1)
                aligned_features.append(feature)
                continue
                
            split_indices = (cot_ids == newline_id).nonzero(as_tuple=True)[0].tolist()
            cot_seq_len = cot_ids.shape[0]
            
            if not split_indices or split_indices[-1] != cot_seq_len - 1:
                split_indices.append(cot_seq_len - 1)

            segment_reps = [cot_reps[end_idx, :] for end_idx in split_indices]
            segment_stack = torch.stack(segment_reps)
            num_teacher_segments = segment_stack.size(0)

            if num_teacher_segments >= num_thinking_steps:
                indices = torch.linspace(0, num_teacher_segments - 1, num_thinking_steps).long()
                aligned_rep = segment_stack[indices]
            else:
                last_rep = segment_stack[-1].unsqueeze(0)
                repeats = num_thinking_steps - num_teacher_segments
                aligned_rep = torch.cat([segment_stack, last_rep.repeat(repeats, 1)], dim=0)
                
            aligned_features.append(aligned_rep)

        aligned_features = torch.stack(aligned_features)
        return reps, aligned_features

    def encode_query(
        self, 
        qry: Dict[str, torch.Tensor], 
        num_thinking_steps: int, 
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the query with simulated thinking steps."""
        num_thinking_steps -= 1
        
        input_ids = qry['input_ids'].clone()
        attention_mask = qry['attention_mask'].clone()
        batch_size = input_ids.shape[0]

        if num_thinking_steps > 0:
            eos_padding = torch.full((batch_size, num_thinking_steps), self.eos_token_id, dtype=torch.long, device=input_ids.device)
            mask_padding = torch.ones((batch_size, num_thinking_steps), dtype=torch.long, device=attention_mask.device)
            
            input_ids = torch.cat([input_ids, eos_padding], dim=1)
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        seq_length = input_ids.shape[1]
        max_query_length = seq_length - num_thinking_steps

        query_embeddings = self.encoder.get_input_embeddings()(input_ids)
        embedding_table = self.encoder.get_input_embeddings().weight
        
        past_key_values = None
        collect_last_hiddens = []

        for step_idx in range(num_thinking_steps):
            current_use_idx = max_query_length + step_idx
            
            if past_key_values is None:
                input_query_embeddings = query_embeddings[:, :current_use_idx]
                input_attention_mask = attention_mask[:, :current_use_idx]
                outputs = self.encoder(
                    inputs_embeds=input_query_embeddings,
                    attention_mask=input_attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True
                )
            else:
                input_query_embeddings = query_embeddings[:, current_use_idx - 1 : current_use_idx]
                input_attention_mask = attention_mask[:, :current_use_idx]
                outputs = self.encoder(
                    inputs_embeds=input_query_embeddings,
                    attention_mask=input_attention_mask,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )

            last_token_logits = outputs.logits[:, -1, :]
            hidden_states = [layer_item[:, -1, :] for layer_item in outputs.hidden_states]
            collect_last_hiddens.append(hidden_states)
            
            weights = F.softmax(last_token_logits, dim=-1)
            new_thought_embedding = torch.matmul(weights, embedding_table)
            
            past_key_values = outputs.past_key_values

            pre_embeddings = query_embeddings[:, :current_use_idx, :]
            new_embedding = new_thought_embedding.unsqueeze(1)
            post_embeddings = query_embeddings[:, current_use_idx + 1 :, :]
            query_embeddings = torch.cat([pre_embeddings, new_embedding, post_embeddings], dim=1)

        if past_key_values is None:
            outputs = self.encoder(
                inputs_embeds=query_embeddings,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            final_token_embedding = query_embeddings[:, -1:]
            outputs = self.encoder(
                inputs_embeds=final_token_embedding,
                attention_mask=attention_mask,
                output_hidden_states=True,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True
            )

        reps = [layer_item[:, -1, :] for layer_item in outputs.hidden_states]
        collect_last_hiddens.append(reps)

        final_reps = torch.stack([item[-1] for item in collect_last_hiddens], dim=1)
        final_reps = torch.mean(final_reps, dim=1)
        final_reps = F.normalize(final_reps, p=2, dim=-1)

        all_hiddens = torch.stack([item[-1] for item in collect_last_hiddens], dim=1)

        return final_reps, all_hiddens

    def encode_passage(
        self, 
        qry: Dict[str, torch.Tensor], 
        num_thinking_steps: int, 
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_query(qry, num_thinking_steps, is_inference)

    def forward(
        self, 
        query: Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None, 
        passage: Dict[str, torch.Tensor] = None, 
        current_step: Optional[int] = None,
        ref: Optional[Dict[str, torch.Tensor]] = None, 
        ref_labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> EncoderOutput:
        num_thinking_steps = self.num_thinking_steps
        
        if isinstance(query, (tuple, list)) and len(query) == 2:
            query_dict, rewrite_query_dict = query
        else:
            query_dict = query
            rewrite_query_dict = None

        q_reps, q_hiddens = self.encode_query(query_dict, num_thinking_steps, is_inference=False)
        p_reps, p_hiddens = self.encode_passage(passage, num_thinking_steps, is_inference=False)
        
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        if self.training:
            if rewrite_query_dict is None:
                raise ValueError("Training requires a tuple of (query, rewrite_query) for distillation.")
                
            teacher_q, teacher_middle = self.encode_teacher(rewrite_query_dict, query_dict['attention_mask'], num_thinking_steps)

            if getattr(self, "is_ddp", False):
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                teacher_q = self._dist_gather_tensor(teacher_q)
                q_hiddens = self._dist_gather_tensor(q_hiddens)
                teacher_middle = self._dist_gather_tensor(teacher_middle)

            scores = self.compute_similarity(q_reps, p_reps, is_inference=True)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(q_reps.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            q_d_loss = self.compute_loss(scores / self.temperature, target)

            teacher_scores = self.compute_similarity(teacher_q, p_reps, is_inference=True)
            teacher_scores = teacher_scores.view(teacher_q.size(0), -1)
            loss_kl = self.compute_kl_loss(scores, teacher_scores.detach(), self.temperature)

            loss_teacher = self.compute_loss(teacher_scores / self.temperature, target)

            student_steps_flat = F.normalize(q_hiddens.view(-1, q_hiddens.size(-1)), p=2, dim=-1)
            teacher_steps_flat = F.normalize(teacher_middle.view(-1, teacher_middle.size(-1)), p=2, dim=-1)
            
            step_student_scores = torch.matmul(student_steps_flat, p_reps.transpose(0, 1))
            step_teacher_scores = torch.matmul(teacher_steps_flat, p_reps.transpose(0, 1))

            loss_latent = self.compute_kl_loss(step_student_scores, step_teacher_scores.detach(), self.latent_temperature)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.info(
                    f"Losses - q_d_loss: {q_d_loss.item():.4f}, loss_kl: {loss_kl.item():.4f}, "
                    f"loss_teacher: {loss_teacher.item():.4f}, loss_latent: {loss_latent.item():.4f}"
                )

            loss = q_d_loss + (self.kl_weight * loss_kl) + loss_teacher + (self.latent_weight * loss_latent)

            if getattr(self, "is_ddp", False):
                loss = loss * self.world_size  

        else:
            scores = self.compute_similarity(q_reps, p_reps, is_inference=True)
            loss = None
            
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_similarity(self, q_reps: torch.Tensor, p_reps: torch.Tensor, is_inference: bool = False) -> torch.Tensor:
        if is_inference:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.einsum('bnd,md->bnm', q_reps, p_reps)

    def compute_loss(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(scores, target)

    def compute_kl_loss(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        temp = temperature if temperature is not None else 1.0
        p_target = F.softmax(teacher_scores / temp, dim=-1)
        p_student = F.log_softmax(student_scores / temp, dim=-1)
        return F.kl_div(p_student, p_target, reduction='batchmean')

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable()

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if getattr(model_args, "lora", False) or getattr(model_args, "lora_name_or_path", None):
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
                
            if getattr(model_args, "lora_name_or_path", None):
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.CAUSAL_LM,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
                
            model = cls(
                encoder=lora_model,
                tokenizer=tokenizer,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                num_thinking_steps=model_args.num_thinking_steps,
            )
        else:
            model = cls(
                encoder=base_model,
                tokenizer=tokenizer,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                num_thinking_steps=model_args.num_thinking_steps
            )
        return model

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        pooling: str = 'cls',
        normalize: bool = False,
        lora_name_or_path: Optional[str] = None,
        num_thinking_steps: int = 3,
        **hf_kwargs
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        base_model.train()
        
        for param in base_model.parameters():
            param.requires_grad = True
            
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            encoder_model = lora_model
        else:
            encoder_model = base_model

        model = cls(
            encoder=encoder_model,
            tokenizer=tokenizer,
            pooling=pooling,
            normalize=normalize,
            num_thinking_steps=num_thinking_steps,
        )
        return model
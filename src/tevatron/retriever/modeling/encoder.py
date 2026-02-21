from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, current_step=None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def _dist_all_reduce_mean(self, t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if not self.is_ddp:
            return t
        # 统一为张量、float32、与设备一致，且不带梯度
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t.view(1)
        t = t.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        # 同步到这一点，避免顺序不一致
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / self.world_size
        # 再同步一次，避免后续不一致
        dist.barrier()
        return t.squeeze(0)

    # def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
    #     if t is None:
    #         return None
    #     # 支持标量张量（0维）和Python标量
    #     if not isinstance(t, torch.Tensor):
    #         t = torch.tensor(t, device=self.encoder.device if hasattr(self.encoder, 'device') else None)

    #     is_scalar = (t.dim() == 0)
    #     if is_scalar:
    #         # 将0维张量转换为形状为[1]的一维张量，便于cat聚合
    #         t = t.view(1)

    #     t = t.contiguous()

    #     all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
    #     dist.all_gather(all_tensors, t)

    #     # 保证本rank的数据是正确的（避免NCCL未覆盖的情况）
    #     all_tensors[self.process_rank] = t
    #     gathered = torch.cat(all_tensors, dim=0)

    #     # 对标量输入，返回形状为[world_size]的一维张量
    #     return gathered

    # def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
    #     if t is None:
    #         return None
    #     t = t.contiguous()

    #     all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
    #     dist.all_gather(all_tensors, t)

    #     all_tensors[self.process_rank] = t
    #     all_tensors = torch.cat(all_tensors, dim=0)

    #     return all_tensors

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        """
        Gathers a tensor from all processes in a distributed setup.
        Handles both regular tensors and scalar tensors correctly.
        """
        if t is None:
            return None
        
        # 确保张量在内存中是连续的，这是很多后端通信操作的要求
        t = t.contiguous()

        # 1. 准备接收列表
        # 为每个进程创建一个形状、类型和设备都与输入张量t相同的空张量
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]

        # 2. 执行 all_gather 操作
        # all_gather 会将每个进程上的 t 收集到 all_tensors 列表中
        # 例如，来自 process i 的 t 会被放到 all_tensors[i]
        dist.all_gather(all_tensors, t)

        # 注意：下面这行是多余的，可以删除。
        # all_gather 操作已经将当前进程的 t 填充到了 all_tensors[self.process_rank] 中。
        all_tensors[self.process_rank] = t 

        # 3. 拼接张量 (核心修改)
        # 检查输入的张量t是否是标量 (0维)
        if t.dim() == 0:
            # 如果是标量，使用 torch.stack 创建一个新的维度并堆叠
            # [tensor(1.0), tensor(2.0)] -> tensor([1.0, 2.0])
            gathered_tensors = torch.stack(all_tensors, dim=0)
        else:
            # 如果不是标量，使用 torch.cat 沿着第0个维度 (通常是batch维度) 拼接
            # [torch.rand(2, 3), torch.rand(2, 3)] -> torch.rand(4, 3)
            gathered_tensors = torch.cat(all_tensors, dim=0)

        return gathered_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **hf_kwargs)
        if 'ouro' in model_args.model_name_or_path.lower():
            base_model.total_ut_steps = model_args.ouro_step
            print("success set ouro step:", model_args.ouro_step)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        model.cuda()
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

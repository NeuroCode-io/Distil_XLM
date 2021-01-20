import math
import os
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Distiller:
    def __init__(self, params, dataloader, student, teacher, device):
        # Initializing Distiller
        self.params = params
        self.dump_path = params["dump_path"]
        self.student = student
        self.teacher = teacher
        self.device = device
        self.dataloader = dataloader

        self.temperature = params["temperature"]
        assert self.temperature > 0.0

        self.alpha_ce = params["alpha_ce"]
        self.alpha_mlm = params["alpha_mlm"]
        self.alpha_mse = params["alpha_mse"]
        self.alpha_cos = params["alpha_cos"]

        self.mlm_mask_prop = params["mlm_mask_prop"]
        assert 0.0 <= self.mlm_mask_prop <= 1.0

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0

        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        #  Initializing model optimizer
        assert params["gradient_accumulation_steps"] >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(
                self.num_steps_epoch
                / params["gradient_accumulation_steps"]
                * params["n_epoch"]
            )
            + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=params["learning_rate"],
            eps=params["adam_epsilon"],
            betas=(0.9, 0.98),
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params["warmup_prop"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

    def train(self):
        """
        The real training loop.
        """
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params["n_epoch"]):
            iter_bar = tqdm(self.dataloader, desc="-Iter")
            for batch in iter_bar:
                # batch = tuple(t.to(device) for t in batch)
                b_input_ids = batch["input_ids"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                b_bool_attn_mask = batch["input_ids"] != 0
                b_bool_attn_mask.to(self.device)

                self.step(
                    input_ids=b_input_ids,
                    attention_mask=b_bool_attn_mask,
                    lm_labels=b_labels,
                )

                iter_bar.update()
            iter_bar.close()
            self.end_epoch()

        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        print("Training is finished")

    def step(self, input_ids, attention_mask, lm_labels):
        s_output = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )  # (bs, seq_length, voc_size)
        s_logits, s_hidden_states = s_output["logits"], s_output["hidden_states"]
        with torch.no_grad():
            t_output = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            t_logits, t_hidden_states = t_output["logits"], t_output["hidden_states"]

        assert s_logits.size() == t_logits.size()

        mask = (
            (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)
        )  # (bs, seq_length, voc_size)
        # or  mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(
            s_logits, mask
        )  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(
            -1, s_logits.size(-1)
        )  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(
            t_logits, mask
        )  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(
            -1, s_logits.size(-1)
        )  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(
                s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1)
            )
            loss += self.alpha_mlm * loss_mlm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(
                s_logits_slct, t_logits_slct
            ) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(
                s_hidden_states
            )  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(
                s_hidden_states, mask
            )  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(
                -1, dim
            )  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(
                t_hidden_states, mask
            )  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(
                -1, dim
            )  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(
                1
            )  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(
                s_hidden_states_slct, t_hidden_states_slct, target
            )
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            print("NaN detected")
            exit()

        if self.params["gradient_accumulation_steps"] > 1:
            loss = loss / self.params["gradient_accumulation_steps"]

        loss.backward()
        self.iter()
        if self.n_iter % self.params["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.params["max_grad_norm"]
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        mdl_to_save = (
            self.student.module if hasattr(self.student, "module") else self.student
        )
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))

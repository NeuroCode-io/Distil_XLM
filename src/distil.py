import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, DistilBertForMaskedLM
from transformers.data.data_collator import DataCollatorForLanguageModeling

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch

from dataset import LanguageModelingDataset
from distiller import Distiller



# Getting dataset
df = pd.read_csv("./data/SST-2/train.tsv", encoding='utf-8', sep='\t')
# len df is 67.349, since we are working on cpu we only take 3000
train_df = df.iloc[:3000]

# Getting teachers tokenizer and preparing data
teacher_model_name = "bert-base-uncased"
student_model_name = "distilbert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(teacher_model_name)

dataset = LanguageModelingDataset(train_df["sentence"], teacher_model_name, sort=False)
collate_fn = DataCollatorForLanguageModeling(tokenizer)
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=32)

# Getting teacher and student model
teacher = BertForMaskedLM.from_pretrained(teacher_model_name)
student = DistilBertForMaskedLM.from_pretrained(student_model_name)

# needed paramteres for training
params = {"n_epoch": 3,"temperature": 2.0, "alpha_ce": 0.5, "alpha_mlm": 2.0, "alpha_cos": 1.0, "alpha_mse": 1.0, "gradient_accumulation_steps": 50, "learning_rate": 5e-4, "adam_epsilon": 1e-6, "weight_decay": 0.0, "warmup_prop": 0.05, "max_grad_norm": 5.0, "dump_path": "output", "mlm_mask_prop": 0.15}
device = torch.device("cpu")

# Initializing Distiller class
distiller = Distiller(params, dataloader, student, teacher, device)

# train
distiller.train()
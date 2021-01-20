import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, DistilBertForMaskedLM
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from dataset import LanguageModelingDataset
from transformers.data.data_collator import DataCollatorForLanguageModeling


# Getting dataset
df = pd.read_csv("./data/SST-2/train.tsv", encoding='utf-8', sep='\t')
# len df is 67.349, since we are working on cpu we only take 3000
train_df = df.iloc[:3000]

# Getting teachers tokenizer and preparing data
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset = LanguageModelingDataset(train_df["sentence"], model_name, sort=False)
collate_fn = DataCollatorForLanguageModeling(tokenizer)
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=32)

# Getting teacher and student model
teacher = BertForMaskedLM.from_pretrained(model_name)
student = DistilBertForMaskedLM.from_pretrained(model_name)
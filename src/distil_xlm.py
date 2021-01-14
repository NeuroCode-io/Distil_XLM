import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import XLMForSequenceClassification, XLMTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer

from xlm_data import df_to_Tensordataset
from utils import device, set_seed, batch_to_inputs, distillation_settings, TRAIN_FILE, ROOT_DATA_PATH


set_seed(3)

# Get data 
train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')

# Get teacher model
xlm_model = XLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048')
xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

# Get datasets and dataloaders
data_set = df_to_Tensordataset(train_df, xlm_tokenizer)
sampler = SequentialSampler(data_set)
data = DataLoader(data_set, sampler=sampler, batch_size=distillation_settings['train_batch_size'])

xlm_model.to(device())
xlm_model.eval()

# Get XLM predictions 
xlm_logits = None

for batch in tqdm(data, desc="xlm logits"):
    batch = tuple(t.to(device()) for t in batch)
    inputs = batch_to_inputs(batch)

    with torch.no_grad():
        outputs = xlm_model(**inputs)
        _, logits = outputs[:2]

        logits = logits.cpu().numpy()
        if xlm_logits is None:
            xlm_logits = logits
        else:
            xlm_logits = np.vstack((xlm_logits, logits))

# Fine-tune DistilmBERT! 

# Define datasets for student model
X_train = train_df['sentence'].values
y_train = xlm_logits
y_real = train_df['label'].values

# 
student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
student_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
student_model.to(device)

# getting troch.tensors for xlm_logits
tensor_xlm_logits = torch.tensor(xlm_logits)

# 4. train
distiller.train(X_train, y_train, y_real, ROOT_DATA_PATH) 


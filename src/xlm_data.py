import pandas as pd
from torch.utils.data import TensorDataset
import torch 


def df_to_Tensordataset(df, tokenizer):
    # encode to obatin input_ids and attention_masks
    encoded = tokenizer(df.sentence.values.tolist(), padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']

    # getting labels
    labels = df.label.values

    # get torch.tensors
    tensor_inputs = torch.tensor(input_ids)
    tensor_labels = torch.tensor(labels)

    tensor_masks = torch.tensor(attention_masks)

    return TensorDataset(tensor_inputs, tensor_masks, tensor_labels)

def student_Tensordataset(df, y_real, tokenizer):
    encoded = tokenizer(df.sentence.values.tolist(), padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    labels = df.label.values

    # get torch.tensors
    tensor_inputs = torch.tensor(input_ids)
    tensor_labels = torch.tensor(labels)
    tensor_y_real = torch.tensor(y_real)
    return TensorDataset(tensor_inputs, tensor_labels, tensor_y_real)




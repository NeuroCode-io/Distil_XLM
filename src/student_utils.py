import torch
from torch import nn

class Student(nn.Module):

    def __init__(self, settings, model):
        self.settings = settings
        self.model = model 
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.a = 0.5

    def loss(self, output, xlm_prob, real_label):
        return self.a * self.criterion_ce(output, real_label) + (1 - self.a) * self.criterion_mse(output, xlm_prob)

    @staticmethod
    def to_device(text, xlm_prob, real_label):
        text = text.to(device())
        xlm_prob = xlm_prob.to(device())
        real_label = real_label.to(device())
        return text, xlm_prob, real_label

    @staticmethod
    def optimizer(model):
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return optimizer, scheduler

    def train(self, model, data_set, output_dir):
        train_settings = self.settings
        num_train_epochs = train_settings['num_train_epochs']

        for epoch in range(num_train_epochs):
            train_loss = 0
            train_sampler = RandomSampler(data_set)
            data_loader = DataLoader(data_set, sampler=train_sampler, batch_size=self.settings['train_batch_size'], drop_last=True)

            model.train()
            optimizer, scheduler = self.optimizer(model)
            for i, (text, xlm_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):
                text, xlm_prob, real_label = self.to_device(text, xlm_prob, real_label)
                model.zero_grad()
                output = model(text.t()).squeeze(1)
                loss = self.loss(output, xlm_prob, real_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
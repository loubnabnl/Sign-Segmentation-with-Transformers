import os
import time
import math
import copy
import pickle
import json
from math import ceil
from pathlib import Path
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import Bar
from utils.viz import viz_results_paper
from utils.averagemeter import AverageMeter
from utils.utils import torch_to_list, get_num_signs
from eval import Metric


class TransformerModel(nn.Module):
    def __init__(self, nhead, dim_feedforward, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        dim_feedforward : the feedforward dimension of the model. 
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        ##self.encoder = nn.Embedding(ntoken, nhid) # fill me, nhid = the dim_embed
        self.pos_encoder = PositionalEncoding(nhid) #fill me, the PositionalEncoding class is implemented in the next cell
        encoder_layers = nn.TransformerEncoderLayer(nhid, nhead, dim_feedforward=dim_feedforward) #fill me we assume nhid = d_model = dim_feedforward
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) #fill me
        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        ##self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask):
        ## !! to do : padding mask 

        ##src = self.encoder(src) * math.sqrt(self.nhid) 
        src = self.pos_encoder(src * math.sqrt(self.nhid))
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class ClassificationHead(nn.Module):
    def __init__(self, nhid, nclasses):
        super(ClassificationHead, self).__init__()
        self.decoder = nn.Linear(nhid , nclasses)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.decoder(src)
        return output
    
class TransformerClassifier(nn.Module):
    def __init__(self, nhead, nhid, dim_feedforward, nlayers, nclasses, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.base = TransformerModel(nhead, dim_feedforward, nhid, nlayers)
        self.classifier =  ClassificationHead(nhid, nclasses)

    def forward(self, src, src_mask, src_key_padding_mask):
        ## !!!! TO ADD : src_key_paddins_mask in transformer !! 

        # base model
        x = self.base(src, src_mask, src_key_padding_mask)
        # classifier model
        output = self.classifier(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransfromerTrainer:
    def __init__(self, nhead, nhid, dim_feedforward, num_layers, num_classes, dropout, device, weights, save_dir):
        #self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.model = TransformerClassifier(nhead, nhid, dim_feedforward, num_layers, num_classes, dropout)
        self.nhid = nhid
        
        if weights is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=-100)

        self.mse = nn.MSELoss(reduction='none')
        self.mse_red = nn.MSELoss(reduction='mean')
        self.sm = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.writer = SummaryWriter(log_dir=f'{save_dir}/logs')
        self.global_counter = 0
        self.train_result_dict = {}
        self.test_result_dict = {}

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, eval_args, lr_mul=1, n_warmup_steps=100, pretrained='',):
        self.model.train()
        self.model.to(device)

        # load pretrained model
        if pretrained != '':
            pretrained_dict = torch.load(pretrained)
            self.model.load_state_dict(pretrained_dict)

        #optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        #optimizer from Attention is all you need paper
        optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            lr_mul, self.nhid, n_warmup_steps)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            end = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            bar = Bar("E%d" % (epoch + 1), max=batch_gen.get_max_index())
            count = 0
            get_metrics_train = Metric('train')

            while batch_gen.has_next():
                self.global_counter += 1
                batch_input, batch_target, batch_target_eval, padding_mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_target_eval, padding_mask = batch_input.permute(2, 0, 1).to(device), batch_target.permute(1,0).to(device), batch_target_eval.permute(1,0).to(device),  padding_mask.permute(2, 0, 1).to(device)
                #src_mask = self.model.base.generate_square_subsequent_mask(batch_input.size(0)).to(device) ## to change...
                src_mask = None                
                optimizer.zero_grad()
                key_padding_mask = (padding_mask[:,:,0:1]< 1).squeeze(2).permute(1,0)
                predictions = self.model(batch_input, src_mask, key_padding_mask)

                loss = 0
                # loss for each stage !
                loss += self.ce(predictions.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.reshape(-1))
                loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions[1:, :, :], dim=2), F.log_softmax(predictions.detach()[:-1, :, :], dim=2)), min=0, max=16)*padding_mask[1:, :, :])                
                epoch_loss += loss.item()
                loss.backward()
                #optimizer.step()
                optimizer.step_and_update_lr()
                
                _, predicted = torch.max(predictions.data, 2)
                gt = batch_target
                gt_eval = batch_target_eval

                get_metrics_train.calc_scores_per_batch(predicted, gt, gt_eval, padding_mask.permute(0, 2, 1))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = "({batch}/{size}) Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | LR: {lr:} | Loss: {loss:}".format(
                    batch=count + 1,
                    size=batch_gen.get_max_index() / batch_size,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=datetime.timedelta(seconds=ceil((bar.eta_td/batch_size).total_seconds())),
                    lr = round(optimizer._optimizer.param_groups[0]['lr'], 7),
                    loss=loss.item()
                )
                count += 1
                bar.next()

                #print('batch ok !')
                if count % 50 == 0:
                    print(bar.suffix)
            print('epoch ok')
            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            get_metrics_train.calc_metrics()
            result_dict = get_metrics_train.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/(len(batch_gen.list_of_examples)/batch_size))
            self.train_result_dict.update(result_dict)

            eval_args[7] = epoch
            eval_args[1] = save_dir + "/epoch-" + str(epoch+1) + ".model"
            #self.predict(*eval_args)

        #with open(f'{save_dir}/train_results.json', 'w') as fp:
        #    json.dump(self.train_result_dict, fp, indent=4)
        #with open(f'{save_dir}/eval_results.json', 'w') as fp:
        #    json.dump(self.test_result_dict, fp, indent=4)
        #self.writer.close()

    #### TODO :  ADAPT PREDICT FROM model.py : 
    def predict(): 
        pass
    
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling
    source code: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    def state_dict(self):
        ''' Optimizer state '''
        return self._optimizer.state_dict()
import itertools

import torch
import webdataset as wds



class State:
    def __init__(self,ratio,total_batch_size):
        self.ratio=ratio
        self.total_batch_size=total_batch_size
        self.buffer_syn_x=[]
        self.buffer_syn_y = []
        self.buffer_x = []
        self.buffer_y = []

    def update_mix_ratio(self):
        pass

    def get_data(self,origin_dataloader_iter,syn_dataloader_iter):
        dataset_mix_ratio = self.ratio
        total_batch_size = self.total_batch_size

        syn_batch_size = int(total_batch_size * dataset_mix_ratio)
        origin_batch_size = total_batch_size - syn_batch_size

        if len(self.buffer_syn_x)<syn_batch_size:
            x,y=next(syn_dataloader_iter)
            self.buffer_syn_x.append(x)
            self.buffer_syn_y.append(y)

        if len(self.buffer_x)<origin_batch_size:
            x,y=next(origin_dataloader_iter)
            self.buffer_x.append(x)
            self.buffer_y.append(y)

        syn_x,self.buffer_syn_x=self.buffer_syn_x[:syn_batch_size],self.buffer_syn_x[syn_batch_size:]
        syn_y, self.buffer_syn_y = self.buffer_syn_y[:syn_batch_size], self.buffer_syn_y[syn_batch_size:]

        x, self.buffer_x = self.buffer_x[:origin_batch_size], self.buffer_x[origin_batch_size:]
        y, self.buffer_y = self.buffer_y[:origin_batch_size], self.buffer_y[origin_batch_size:]

        x=torch.cat([syn_x,x])
        y=torch.cat(syn_y,y)
        return x,y






s=State(0.9,4096)
x=torch.ones(64,3,128,128)+2
y=torch.ones(64,100)
s.buffer.extend(x)

print(torch.stack(s.buffer).shape)
# x,y=s.buffer[:64]
# print(x.shape,y.shape)
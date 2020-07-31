# -*- coding: utf-8 -*- 
import os
import numpy as np
import torch



# save network
def save(ckpt_dir, net, optim, epoch):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  
  torch.save({
      'net': net.state_dict(), 'optim' : optim.state_dict()},
      "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
  

# load network
def load(ckpt_dir, net, optim, load_opt=None):
  if not os.path.exists(ckpt_dir):
    epoch = 0
    return net, optim, epoch
  
  if load_opt is None:
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 숫자만 이용하여 소팅
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]))
  else:
      epoch = int(load_opt)
      dict_model = torch.load('%s/model_epoch%d.pth' %(ckpt_dir, epoch))

  net.load_state_dict(dict_model['net'])
  optim.load_state_dict(dict_model['optim'])

  return net, optim, epoch


def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory. ' + dir)

def adjust_lr(optim, init_lr, epoch, lr_decay_epoch, max_epoch, power=0.1):
    if epoch % lr_decay_epoch or epoch > max_epoch:
        return optim
    lr = init_lr*(power ** (epoch // lr_decay_epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    


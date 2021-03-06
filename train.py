# -*- coding: utf-8 -*- 
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from model import *
from dataset import *
from util import *
import torchvision
from torchvision import transforms
from datetime import datetime
import torchnet as tnt
from sklearn.metrics import accuracy_score
from mobilenetv2.models.imagenet import mobilenetv2

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ## Parameter
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    load_opt = args.load_opt
    date = args.date
    
    mode = args.mode
    train_continue = args.train_continue

    #task = args.task
    #opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    #nker = args.nker

    network = args.network
    #learning_type = args.learning_type
    if date is None:
        now = datetime.now()
        date = now.strftime('%Y-%m-%d(%H:%M)')
    #date = "2020-07-10(12:53)"
    

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir + "/" + network + date
    log_dir = args.log_dir + "/" + network + date
    result_dir = args.result_dir + "/" + network + date

    # random seed
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False


    # 네트워크 생성
    if network == "resnet":
        #net = ResNet(in_channels=nch, out_channels=nch, norm="bnorm").to(device)
        #net = torchvision.models.resnet50(pretrained=True)
        net = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        #for param in net.parameters():
        #    param.requires_grad=False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net = net.to(device)
        params = net.parameters()
    elif network == "mobilenet":
        #net = torchvision.models.mobilenet_v2(pretrained=True)
        net = mobilenetv2(width_mult=0.5)
        net.load_state_dict(torch.load('./mobilenetv2/pretrained/mobilenetv2_0.5-eaa6f9ad.pth'))
        net.classifier = nn.Linear(net.output_channel, 2)
        net = net.to(device)

        #net = mobilenet_v2(pretrained=True, input_size=11).to(device)
        #net.classifier = nn.Linear(net.last_channel, 2)
        #net = net.to(device)
        params = net.parameters()
    elif network == "efficientnet":
        net = EfficientNet.from_pretrained('efficientnet-b0')
        net._fc = nn.Linear(in_features=net._fc.in_features, out_features=2, bias=True)
        params = net.parameters()
        net = net.to(device)


    # 로스함수 정의
    #fn_loss = nn.BCEWithLogitsLoss().to(device)
    fn_loss = nn.CrossEntropyLoss().to(device)

    # optimizer 정의
    optim = torch.optim.Adam(params, lr=lr)
    #optim = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.001)

    createFolder(os.path.join(log_dir, 'train'))
    createFolder(os.path.join(log_dir, 'val'))
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    #writer_train_acc = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    #writer_val_acc = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)


    if mode=='train':
        # 네트워크 학습하기
        transform_train = transforms.Compose([transforms.Resize([ny,nx]),  
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        transform_val = transforms.Compose([transforms.Resize([ny,nx]),  
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        dataset_train = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

        dataset_val = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val / batch_size)





    if mode=='train':
        ############# Training ################
        if train_continue=="on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim, load_opt=load_opt)
        else:
            st_epoch = 0 # start epoch number



        for epoch in range(st_epoch+1, num_epoch + 1):
            
            #adjust_lr(optim=optim, init_lr=lr, epoch=epoch, lr_decay_epoch=5, max_epoch=num_epoch, power=0.1)
            

            net.train()
            #loss_train = []
            # torchnet meter
            loss_meter_T = tnt.meter.AverageValueMeter()
            acc_meter_T = tnt.meter.ClassErrorMeter(accuracy=True)
            

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                
                input, label = data
                label = label.to(device)
                input = input.to(device)

                optim.zero_grad()
                
                output = net.forward(input)
                #_, preds = torch.max(output, 1)
                
                loss = fn_loss(output.squeeze(), label)
                loss.backward()
                optim.step()
                # loss function 계산
                #loss_train += [loss.item()]
                loss_meter_T.add(loss.item())
                acc_meter_T.add(output.data.cpu().numpy(), label.cpu().numpy().squeeze())
                
                if batch % 500 == 0 or batch == num_batch_train:
                    print("TRAIN : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % 
                        (epoch, num_epoch, batch, num_batch_train, loss_meter_T.value()[0], acc_meter_T.value()[0]))
             
            scheduler.step()

            writer_train.add_scalar('loss', loss_meter_T.value()[0], epoch)
            writer_train.add_scalar('accuracy', acc_meter_T.value()[0], epoch)


        


            ############### validation ###############
            with torch.no_grad():
                net.eval()
                #loss_val = []
                loss_meter_V = tnt.meter.AverageValueMeter()
                acc_meter_V = tnt.meter.ClassErrorMeter(accuracy=True)

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    input, label = data
                    label = label.to(device)
                    input = input.to(device)

                    output = net.forward(input)

                    loss = fn_loss(output.squeeze(), label)
                    #loss_val += [loss.item()]
                    loss_meter_V.add(loss.item())
                    acc_meter_V.add(output.data.cpu().numpy(), label.cpu().numpy().squeeze())

                    
                print("VALID : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f\n" % 
                    (epoch, num_epoch, batch, num_batch_val, loss_meter_V.value()[0], acc_meter_V.value()[0]))

                    
                writer_val.add_scalar('loss', loss_meter_V.value()[0], epoch)
                writer_val.add_scalar("accuracy", acc_meter_V.value()[0], epoch)

                
            if epoch % 1 == 0 or epoch == num_epoch:
                createFolder(ckpt_dir)
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        writer_train.close()
        writer_val.close()
        


          
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Parameter
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    load_opt = args.load_opt
     

    
    mode = args.mode
    train_continue = args.train_continue

    #task = args.task
    #opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    #nker = args.nker

    network = args.network

    
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
   # log_dir = args.log_dir + "/" + network + date
   # result_dir = args.result_dir + "/" + network + date

#    createFolder(result_dir)


    # 로스함수 정의
    #fn_loss = nn.BCEWithLogitsLoss().to(device)
    fn_loss = nn.CrossEntropyLoss().to(device)
    

    if network == "resnet":
        #net = ResNet(in_channels=nch, out_channels=nch, norm="bnorm").to(device)
        #net = torchvision.models.resnet50(pretrained=True)
        net = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        #for param in net.parameters():
        #    param.requires_grad=False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net = net.to(device)
        params = net.parameters()
    elif network == "mobilenet":
        #net = torchvision.models.mobilenet_v2(pretrained=True)
        #net.classifier[1] = nn.Linear(net.last_channel, 2)
        #net = net.to(device)
        #net = mobilenet_v2(pretrained=True, input_size=11).to(device)
        #net.classifier = nn.Linear(net.last_channel, 2)
        #net = net.to(device) 
        net = mobilenetv2(width_mult=0.5)
        net.load_state_dict(torch.load('./mobilenetv2/pretrained/mobilenetv2_0.5-eaa6f9ad.pth'))
        net.classifier = nn.Linear(net.output_channel, 2)
        net = net.to(device)
        params = net.parameters()
    elif network == "efficientnet":
        net = EfficientNet.from_pretrained('efficientnet-b0')
        net._fc = nn.Linear(in_features=net._fc.in_features, out_features=2, bias=True)
        params = net.parameters()
        net = net.to(device)
    
    # optimizer 정의
    optim = torch.optim.Adam(params, lr=lr)


    if mode == "test":
        transform = transforms.Compose([transforms.Resize([ny,nx]),  transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        
        dataset_test = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
        #print(dataset_test.classes)
        # 부수적인 변수들 정의
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)



    if mode == "test":
        ############# Test ################

#        createFolder(os.path.join(result_dir, "female"))
 #       createFolder(os.path.join(result_dir, "male"))
        
        net, optim, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim, load_opt=load_opt)
        net = net.to(device)

        with torch.no_grad():
            net.eval()
            avg_loss = []
            avg_acc = []

            loss_meter_test = tnt.meter.AverageValueMeter()
            acc_meter_test = tnt.meter.ClassErrorMeter(accuracy=True)


            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input, label = data
                label = label.to(device)
                input = input.to(device)

                output = net(input)
                
                loss = fn_loss(output.squeeze(), label)
                loss_meter_test.add(loss.item())
                acc_meter_test.add(output.data.cpu().numpy(), label.cpu().numpy().squeeze())

                avg_loss += [loss_meter_test.value()[0]]
                avg_acc += [acc_meter_test.value()[0]]
                

                print("TEST :  BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % 
                    (batch, num_batch_test, loss_meter_test.value()[0], acc_meter_test.value()[0]))
                
                
                
                
                
        print("\nAVERAGE TEST PERFORMANCE: LOSS %.4f | ACCURACY %.4f" % 
              (np.mean(avg_loss), np.mean(avg_acc)))


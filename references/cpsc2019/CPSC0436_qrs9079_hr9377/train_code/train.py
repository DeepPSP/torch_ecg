import torch
from torch.utils.data import DataLoader
from dataset import QRS_dataset,QRS_val_dataset
from YOLO_1d_HR_4h_loss_ava import Yolo_1d
from optim import Ranger
from utils import *
import numpy as np
np.set_printoptions(suppress=True)
from config import opt

torch.cuda.set_device(opt.gpu_num)


train_dts = QRS_dataset(opt.train_data_list)

train_loader = DataLoader(train_dts,batch_size=opt.train_bs,
                          shuffle=True,
                          num_workers=1)



model = Yolo_1d(c=128).cuda()
optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=1e-4)


with open('log.txt','w') as f:
    for epoch in range(opt.epoch):
        for i,(sig,target) in enumerate(train_loader):
            sig = sig.cuda()
            model.train()
            loss = model(sig,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = get_lr(optimizer)
            print("epoch:{}, lr:{:5f}, batch_{}_loss:{}".format(epoch,lr,i,loss))
        print('--------------------------------------------------')


        if epoch>220:
            torch.save(model.state_dict(), opt.model_dir + 'epoch:%d' % (epoch))


        # f.write('Epoch:{}, lr:{}, QRS_acc:{:6f} \n'.format(epoch,lr,np.mean(rec_accs)))

        # print('--------------------------------------------------')

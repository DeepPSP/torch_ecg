import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
# from torchsummary import summary

def layer_1(in_channel,out_channel):
    layer = nn.Sequential(
        nn.Conv1d(in_channel,out_channel,16,stride=2,padding=7),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(True),
        nn.MaxPool1d(2,2)
    )
    return layer

class GCModule(nn.Module):
    """
    Global Context (GC) block

    the difference with ref. [1,2] lies in the treatment of mid-channels of `channel_att`

    References:
    -----------
    [1] Cao, Yue, et al. "Gcnet: Non-local networks meet squeeze-excitation networks and beyond." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019.
    [2] https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    """
    def __init__(self,channels,reduction=16,mode='mul'):
        super(GCModule,self).__init__()
        self.mode = mode
        mid_channels = channels // reduction
        self.channel_att = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1),
            nn.LayerNorm([mid_channels,1]),
            nn.ReLU(True),
            nn.Conv1d(mid_channels, channels, kernel_size=1),
                )
        self.conv_mask = nn.Conv1d(channels,1,kernel_size=1)
        self.softmax = nn.Softmax(dim=2)


    def sptial_att(self,x):
        input_x = x.unsqueeze(1)
        context = self.conv_mask(x)
        context = self.softmax(context)
        context = torch.matmul(input_x,context.unsqueeze(3))
        return context.squeeze(1)

    def forward(self, x):
        context = self.sptial_att(x)
        att = self.channel_att(context)
        if self.mode == 'add':
            return x + att
        else:
            return x * torch.sigmoid(att)

class Residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,down=False,drop=True):
        super(Residual_block,self).__init__()
        self.down = down
        self.do_drop = drop

        if down:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=7, stride=stride, padding=3, dilation=1),
                nn.BatchNorm1d(out_channel))
            self.conv2 = nn.Sequential(
                nn.Conv1d(out_channel, out_channel, kernel_size=7, stride=1, padding=3, dilation=1),
                nn.BatchNorm1d(out_channel))
            self.down_sample = nn.Sequential(nn.Conv1d(in_channel,out_channel,1,stride=2))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=7, stride=stride, padding=6, dilation=2),
                nn.BatchNorm1d(out_channel))
            self.conv2 = nn.Sequential(
                nn.Conv1d(out_channel, out_channel, kernel_size=7, stride=1, padding=6, dilation=2),
                nn.BatchNorm1d(out_channel))

        self.GC = GCModule(out_channel)
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.GC(x)
        if self.down:
            identity = self.down_sample(identity)
        x += identity
        x = self.relu(x)
        if self.do_drop:
            x = self.drop(x)

        return x

def encoder(labels,region_num):
    N = labels.shape[0]
    target = torch.zeros(N,region_num, 2)
    region_size = 1. / region_num
    # for i in range(len(label)):
    for i,label in enumerate(labels):
        for l in range(len(label)):
            if label[l] == 0:
                continue
            point = label[l]
            start_p = int(np.floor(point / region_size))
            target[i,start_p, 1] = 1
            delta_p = (point / region_size) - start_p
            target[i,start_p, 0] = delta_p
    return target.cuda()
    
class Yolo_layer(nn.Module):
    def __init__(self):
        super(Yolo_layer,self).__init__()
        # self.bs = bs
        self.mse_loss = nn.MSELoss(size_average=True).cuda()
        self.bce_loss = nn.BCELoss(size_average=True).cuda()

    def forward(self, x,target=None):
        if target is not None:
            region_num = x.size(1)
            target = encoder(target,region_num)
            r_mask = target[:, :, 1] > 0    ## 有标注的区间mask
            all_r_pred = x[r_mask].view(-1, 2)
            all_r_target = target[r_mask].view(-1, 2)
            point_loss = self.mse_loss(all_r_pred[:, 0],  ## 只对有标注的区间进行点回归
                                    all_r_target[:, 0])


            conf_loss = self.bce_loss(x[...,1],target[...,1])
            return conf_loss+point_loss

        else:
            x = x.detach().cpu()
            nR = x.size(1)
            offset = torch.arange(nR).view(1, nR).type(torch.FloatTensor)
            x[..., 0] += offset
            x[..., 0] = (x[..., 0] / nR) * 5000
            return x



class final_layer(nn.Module):
    def __init__(self,in_ch):
        super(final_layer,self).__init__()
        self.fuse_conv = nn.Sequential(nn.Conv1d(in_ch,in_ch // 2,7,stride=1,padding=3),
                                          nn.BatchNorm1d(in_ch // 2),
                                          nn.ReLU())
        self.conv_final = nn.Sequential(nn.Conv1d(in_ch // 2,2,1))
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = self.fuse_conv(x)
        x = self.conv_final(x)
        x = F.sigmoid(x)
        x = x.permute(0,2,1)
        return x


class StageModule(nn.Module):
    def __init__(self,stage, out_branches, c):
        super(StageModule,self).__init__()
        self.stage = stage
        # self.num_blocks = num_blocks
        self.out_branches = out_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2**i)
            branch = nn.Sequential(
                Residual_block(w, w),
                Residual_block(w, w),
                Residual_block(w, w),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.out_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j :
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j :
                    if i == 0:
                        # self.fuse_layers[-1].append(nn.Upsample(size=313))
                        self.fuse_layers[-1].append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j),c * (2 ** i),kernel_size=1, stride=1),
                            nn.BatchNorm1d(c * (2 ** i)),
                            nn.Upsample(size=625)
                        ))
                # elif i < j and i == 1:
                    elif i == 1:
                        # self.fuse_layers[-1].append(nn.Upsample(size=157))
                        self.fuse_layers[-1].append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm1d(c * (2 ** i)),
                            nn.Upsample(size=313)
                        ))
                    elif i == 2:
                        self.fuse_layers[-1].append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm1d(c * (2 ** i)),
                            nn.Upsample(size=157)
                        ))

                elif i > j:
                    opts = []

                    if i == j+1:
                        opts.append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j), c * (2 ** i), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** i)),
                        ))
                    elif i == j+2:
                        opts.append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j), c * (2 ** (j+1)), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** (j+1))),
                            nn.ReLU(True),
                            nn.Conv1d(c * (2 ** (j+1)), c * (2 ** (j + 2)), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** (j + 2))),

                        ))
                    elif i == j+3:
                        opts.append(nn.Sequential(
                            nn.Conv1d(c * (2 ** j), c * (2 ** (j+1)), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** (j+1))),
                            nn.ReLU(True),
                            nn.Conv1d(c * (2 ** (j+1)), c * (2 ** (j + 2)), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** (j + 2))),
                            nn.ReLU(True),
                            nn.Conv1d(c * (2 ** (j + 2)), c * (2 ** (j + 3)), kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm1d(c * (2 ** (j + 3))),
                        ))
                    self.fuse_layers[-1].append(nn.Sequential(*opts))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused




class Yolo_1d(nn.Module):
    def __init__(self,c=256):
        super(Yolo_1d,self).__init__()

        self.layer_1 = layer_1(1,64)

        self.layer_2 = self._make_layers(64,128,3)

        self.transition1 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv1d(128, c * (2 ** 1), kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(c * (2 ** 1)),
                nn.ReLU(inplace=True),
            )),
        ])

        self.stage2 = nn.Sequential(StageModule(stage=2,out_branches=2,c=c))

        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv1d(c * (2 ** 1), c * (2 ** 2), kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(c * (2 ** 2)),
                nn.ReLU(inplace=True),
            )),
        ])


        self.stage3 = nn.Sequential(StageModule(stage=3,out_branches=3,c=c),
                                    StageModule(stage=3,out_branches=3,c=c))


        self.transition3 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv1d(c * (2 ** 2), c * (2 ** 3), kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(c * (2 ** 3)),
                nn.ReLU(inplace=True),
            )),
        ])

        self.stage4 = nn.Sequential(StageModule(stage=4,out_branches=4,c=c))


        self.conv_o5 = final_layer(in_ch=1024)
        self.conv_o4 = final_layer(in_ch=512)
        self.conv_o3 = final_layer(in_ch=256)
        self.conv_o2 = final_layer(in_ch=128)


        self.yolo = Yolo_layer()



    def _make_layers(self,in_ch,out_ch,blocks):
        layers = []
        layers.append(Residual_block(in_ch,out_ch,stride=2,down=True))
        for _ in range(1,blocks):
            layers.append(Residual_block(out_ch,out_ch))
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        is_training = target is not None
        x = self.layer_1(x)
        x = self.layer_2(x)
        # x = self.layer_3(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]


        x = self.stage3(x)

        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]

        out2,out3,out4,out5 = self.stage4(x)

        f5 = self.conv_o5(out5)
        f4 = self.conv_o4(out4)
        f3 = self.conv_o3(out3)
        f2 = self.conv_o2(out2)

        if is_training:
            y5 = self.yolo(f5,target)
            y4 = self.yolo(f4,target)
            y3 = self.yolo(f3,target)
            y2 = self.yolo(f2,target)
        else:
            y5 = self.yolo(f5)
            y4 = self.yolo(f4)
            y3 = self.yolo(f3)
            y2 = self.yolo(f2)

        out = [y5,y4,y3,y2]
        # tt = torch.cat(out,1)

        return sum(out) if is_training else torch.cat(out,1)

    def cal_dis(self,sig1,sig2):
        distance = torch.abs(sig1[0]-sig2[:,0])
        return distance
    def predict(self,out,conf_thr):
        out = out.detach().cpu()
        hr_ans = []
        r_ans = []
        for i_sig, pred in enumerate(out):  ## nms
            cut_idx = (pred[..., 0] >= 0.5 * 500) & (pred[..., 0] <= 9.5 * 500)
            pred = pred[cut_idx]
            conf_mask = pred[..., 1] > conf_thr
            pred = pred[conf_mask]
            if not pred.size(0):
                hr_ans.append(math.nan)
                r_ans.append(np.array([]))
                continue
            _, conf_sort_idex = torch.sort(pred[:, 1], descending=True)
            pred = pred[conf_sort_idex]
            max_pred = []
            while pred.size(0):
                max_pred.append(pred[0])
                if len(pred) == 1:
                    break
                dis = self.cal_dis(max_pred[-1], pred[1:])
                pred = pred[1:][dis > 80]
            max_pred = torch.cat(max_pred, 0).view(-1, 2)
            _, point_sort_index = torch.sort(max_pred[:, 0])
            max_pred = np.array(max_pred[point_sort_index])
            r_peak = max_pred[:, 0]
            r_hr = np.array([loc for loc in r_peak if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
            hr = round(60 * 500 / np.mean(np.diff(r_hr)))
            hr_ans.append(hr)
            r_ans.append(r_peak)
        return r_ans, hr_ans
    def _predict(self, out, conf_thr):
        out = out.detach()
        hr_ans = []
        r_ans = []

        for i_sig, pred in enumerate(out):
            conf_mask = pred[..., 1] > conf_thr
            pred = pred[conf_mask]
            if not pred.size(0):
                hr_ans.append(math.nan)
                r_ans.append(np.array([]))
                continue
            _, conf_sort_idex = torch.sort(pred[:, 1], descending=True)
            pred = pred[conf_sort_idex]
            max_pred = []
            while pred.size(0):
                max_pred.append(pred[0])
                if len(pred) == 1:
                    break
                dis = self.cal_dis(max_pred[-1], pred[1:])
                pred = pred[1:][dis > 100]
            max_pred = torch.cat(max_pred, 0).view(-1, 2)
            _, point_sort_index = torch.sort(max_pred[:, 0])
            max_pred = np.array(max_pred[point_sort_index])
            idx = (max_pred[:, 0] >= 0.5 * 500) & (max_pred[:, 0] <= 9.5 * 500)
            qrs = max_pred[idx]
            hr_ans.append(0)
            r_ans.append(qrs)
        return r_ans, hr_ans

# if __name__ == "__main__":
#     t = torch.randn(1,1,5000).cuda()
#     model = Yolo_1d(c=128).cuda()
#     model = torch.nn.DataParallel(model, device_ids=[0, 1])

#     # summary(model,(1,5000))

#     tt = model(t)
#     print(1)

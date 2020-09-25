import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_list",type=str,default='./all.txt',help="data for training")
parser.add_argument("--val_data_list",type=str,default='./all.txt',help="data for validation")
parser.add_argument("--gpu_num",type=int,default=0,help="which gpu to use while traning")
parser.add_argument("--train_bs",type=int,default=16,help="batch_size for training")
parser.add_argument("--val_bs",type=int,default=60,help="batch_size for validation")
parser.add_argument("--val_num_workers",type=int,default=1,help="number workers for validation")
parser.add_argument("--epoch",type=int,default=210)
parser.add_argument("--conf_thres",type=float,default=0.4,help="confidence threshold")
parser.add_argument("--nms_thres",type=int,default=80,help="nms distance threshold")
parser.add_argument("--saving_ckpt", type=bool, default=False, help="whether to save a model while training")
parser.add_argument("--saving_interval", type=int, default=100, help="interval of saving the model weight")
parser.add_argument("--model_dir",type=str,default='./checkpoints/',help="direction for saving models")
parser.add_argument("--saving_thres",type=int,default=100,help="nms distance threshold")
parser.add_argument("--cal_time", type=bool, default=False, help="whether to calculate time when evaluating")
parser.add_argument("--learning_rate_dacay", type=bool, default=True, help="whether to save a model while training")



opt = parser.parse_args()


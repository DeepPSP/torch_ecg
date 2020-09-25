
import ReadData
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np

def Flatten(l):
    return [item for sublist in l for item in sublist]


#len_stat = [len(i) for i in short_data]
#ttt = sum([i > 500 for i in len_stat]) + sum([i < 80 for i in len_stat])
#print(ttt/len(len_stat))
#plt.hist(len_stat, bins=200, range=[80,500])

#plt.plot(my_short_train_data[131])

########## plot long short
#my_short_all_data = my_short_train_data + my_short_val_data
#
##ll = sorted(len_stat, reverse=True)
#for i in range(100):
#    fig = plt.figure()
#    plt.plot(my_short_all_data[len_stat.index(ll[i])])
#    plt.savefig('img/'+str(ll[i])+'.png', bbox_inches='tight')
#    plt.close(fig)


############ plot qrs
#QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
#tmp = Flatten(QRS_data)
# plt.hist(tmp, bins=100, range=[600, 2000])

#Counter


############ plot long
#long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
len_stat = [len(i) for i in long_data]
print(len(len_stat))
print(sum(np.array(len_stat) >= 9000))
print(sum(np.array(len_stat) >= 6000))
print(sum(np.array(len_stat) >= 3000))



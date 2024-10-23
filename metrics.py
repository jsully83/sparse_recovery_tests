
import numpy as np
from einops import repeat, rearrange
from seaborn import heatmap
from matplotlib.pyplot import title, xlabel, ylabel



def similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = np.dot(a, b.T)
    d = np.max(np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
    return (n/d).item(), np.arccos(n/d).item() * (180.0/np.pi)


def euc_distance(a, b):
    a.reshape(-1,1)
    b.reshape(-1,1)
    return np.linalg.norm(a - b, 2, keepdims=True).item()

def sum_of_C(a, m, Np):
    a_ = a.reshape(m*m,Np)
    return np.sum(np.abs(a_),1, keepdims=True)

def visualize(arr, plot_name, cmax, sum=False):
    if sum is True:
       arr = np.sum(arr, 1, keepdims=True)
       heatmap(arr, vmin=0, vmax=cmax)
       xlabel
    else:
        heatmap(arr)
        xlabel('Dictionary Poles')
    
    ylabel('Node Interaction')  
    title(plot_name)

def edge_accuracy(gt, pred):
    a = np.count_nonzero(gt,1) > 0
    b = np.count_nonzero(pred,1) > 0
    
    return np.sum(a==b) / a.shape[0]
    

def running_avg(next, prev, count):
    prev += (next - prev) / count
    return prev



# # Assuming x and y are the outputs from np.histogram
# # x, y = np.histogram(C_pred[1], bins=5)
# threshold = 6.90451162e-02
# cc = np.array(C_pred).reshape(-1, 161)
# # cc = np.load('predictions.npy')
# # cc = np.where(cc > threshold, cc, 0)
# print(np.count_nonzero(cc))
# val, bins = np.histogram(cc, range=(eps, cc.max()))
# # Use sns.histplot to plot the histogram
# # sns.histplot([x,y], bins=y)
# plt.figure(figsize=(10,10))
# plt.hist(cc, bins)
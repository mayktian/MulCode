import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import numpy as np
from mulcode import MulCodeCompressor
import cPickle as pickle

import argparse

def str2bool(arg):
    return arg == 'True'


parser = argparse.ArgumentParser(description='Compress Language Model')
parser.add_argument('--data', type=str, help='data dir to ptb data set')
parser.add_argument('--w1', type=str2bool, help='ptb-large or ptb-small')
parser.add_argument('--lm', type=str, default='',help='data dir to W1')
parser.add_argument('--rate', type=float, default=0.05,help='Target compression rate')
parser.add_argument('--M', type=int, default=58,help='Number of code books')
parser.add_argument('--N', type=int, default=8,help='Number of resacle codebook')
parser.add_argument('--K', type=int, default=12,help='Number of codes')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs')

parser.add_argument('--group_size', type=int, default=4000,help='Group size')
args = parser.parse_args()


if args.w1:
    fn = 'W1'
else:
    fn = 'W2'

def remap(matrix, freq):
    index =  np.argsort(freq)[::-1]
    return matrix[index],index,[freq[i] for i in index]

M,K,N = [args.M,args.K,args.N]
data = args.data
lstm = np.load(args.lm,allow_pickle=True)
freq = np.load(data + "/freqs") #path to frequency
setting_str = "_".join(map(str,[M,K,N]))



if args.w1:
    mat = lstm[0]
else:
    mat = lstm[-2]
print(mat.shape)
dim = mat.shape[1]

if dim > 200:
    fn += '.large'
num_vocab =  mat.shape[0]
freq = [freq[i] for i in range(num_vocab)]

mat_,index,freq = remap(mat,freq)
matrix = mat_

model = MulCodeCompressor(matrix.shape[0],matrix.shape[1],M,K,_TAU=1.0,lr=0.001
                            ,num_compressors=N,group_size=args.group_size,compression_rate=args.rate,freq=freq,word_basis=None) #output to data/W1.shu.top_cut.npy
model.init()
if torch.cuda.is_available():
    model = model.cuda()
_,rec_mat,compression_rate = model._train(matrix,freq,index,64,num_epochs=args.epochs)
output_fn = data + "/compressed_matrix/%s.%s_%.5f.npy" % (fn,setting_str1,compression_rate)
np.save(open(output_fn,'w'),rec_mat)
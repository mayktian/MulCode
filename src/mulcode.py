import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from adaptive_compressor import AdaptiveCompressor,init_weight_variable
import math
class MulCodeCompressor(AdaptiveCompressor):

#     _TAU = 1.0
#     _BATCH_SIZE = 64

    def __init__(self, n_words, dim,n_codebooks, n_centroids,_TAU=1.0,lr=0.0001,num_compressors=1,compression_rate=0.04,group_size=4000,adaptive=True,mul=True,freq=None,min_group_size=16,test_in_epoch=True,use_freq=True,word_basis=None,use_full=False,smooth_r=1.0):
        
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        """
        
        super(MulCodeCompressor,self).__init__(n_words, dim,n_codebooks, n_centroids,_TAU=_TAU,lr=lr,mul=mul,test_in_epoch=test_in_epoch,use_full=use_full)
        self.linear_h = None
        
        self.linear_logit = None
        self.sub_compressors = nn.ModuleList()
        self.num_compressors = num_compressors
        
        self.use_freq = use_freq
        self.codebooks_ = nn.ParameterList()
        self.projs_ = nn.ParameterList()
        self.adaptive = adaptive
        self.compression_rate = compression_rate
            

        if self.adaptive:
            self.min_group_size=  min_group_size
            self.freq =freq
            self.cut_offs = [0]
            while self.cut_offs[-1] + group_size < self.V:
                self.cut_offs.append(self.cut_offs[-1] + group_size)
                
            self.num_groups = len(self.cut_offs)

            code_lens = 0
            self.code_offsets = []
            size = self.M
            self.sizes = []
            ranks = self.resolve_ranks(smooth_r)
            for i in range(self.num_groups):
                self.code_offsets.append(self.M - code_lens)
                if i != self.num_groups -1:   
                    size = self.M / self.num_groups
                else:
                    size = self.M - code_lens
                self.codebooks_.append(init_weight_variable([ size * self.K,ranks[i]]))
                self.projs_.append(init_weight_variable([ranks[i],self.embd_dim]))

                self.sizes.append(size)
                code_lens += self.M / self.num_groups
        else:
            self.codebooks_ = init_weight_variable([ self.M * self.K, self.embd_dim])

        for i in range(num_compressors):
            self.sub_compressors.append(AdaptiveCompressor(self.V,self.embd_dim,self.M,self.K,
                                                            codebooks=None,mul=mul,use_full=self.use_full))
        self.top_word_basis = None
        if word_basis is not None:
            M_,K_,_ = word_basis.shape
            self.top_word_basis = nn.Parameter(Variable(torch.from_numpy(word_basis/100.0)),requires_grad=False).cuda()
            self.sub_compressors.append(AdaptiveCompressor(self.V,self.embd_dim,M_,K_,
                                                            codebooks=self.top_word_basis.view(-1,self.embd_dim),mul=mul))
    def resolve_ranks(self,smooth_r=1.0):
        freq = self.freq
        V = len(freq)
        num = self.V * self.M
        num_params = 0.
        num_params_ = self.V* self.embd_dim * 32 *self.compression_rate
        num_params +=  self.embd_dim * 32*self.num_compressors  
        num_params  += num *math.log(self.K,2) *self.num_compressors
        remaining_num_params = num_params_ - num_params
        start = 0
        step_size  = (len(freq) / self.num_groups)
        freq = [math.pow(f,smooth_r) for f in freq]
        overall_cnts = sum(freq)
        ranks = []
        ratio = 1.0
        for i in range(self.num_groups):
            if i!= self.num_groups-1:
                
                freq_ = sum(freq[self.cut_offs[i]:self.cut_offs[i+1]])
            else:
                freq_ = sum(freq[self.cut_offs[i]:])
            
            M_ = self.M / self.num_groups
            rank = int(((float(freq_)/overall_cnts) * remaining_num_params) / (M_ * self.K + self.embd_dim) * ratio)
            if rank > self.embd_dim:
                ratio = float(self.embd_dim) / rank
                rank = self.embd_dim 
            rank = max(self.min_group_size,rank)
            print "using rank:",rank,freq_,self.cut_offs[i],i
            start =  start + step_size
            ranks.append(rank)
        return ranks
    def quantize(self,quantize_func,bit):
        if self.adaptive:
            for codebook in self.codebooks_:
                codebook_ = codebook.data.cpu().numpy()
                codebook_ = quantize_func(codebook_,bit)
                codebook.data = Variable(torch.from_numpy(codebook_).cuda())
            for proj in self.projs_:
                proj_ = proj.data.cpu().numpy()
                proj_ = quantize_func(proj_,bit)
                proj.data = Variable(torch.from_numpy(proj_).cuda())
        else:
            codebook_ = self.codebooks_.data.cpu().numpy()
            codebook_ = quantize_func(codebook_,bit)
            self.codebooks_.data = Variable(torch.from_numpy(codebook_).cuda())
            
        for i in range(self.num_compressors):
            if self.mul:
                scale = self.sub_compressors[i].scale.data.cpu()[:,0,:].numpy()  
                scale = quantize_func(scale,bit)
                self.sub_compressors[i].scale.data =Variable(torch.from_numpy(scale)[:,None,:].cuda())
        if self.top_word_basis is not None:
            word_basis = self.sub_compressors[-1].codebooks.data.cpu().numpy() 
            word_basis = quantize_func(word_basis,bit)
            self.sub_compressors[-1].codebooks.data =Variable(torch.from_numpy(word_basis).cuda())
            
            
    def compress(self, word_vecs,freq,group_mask):
        """Export the graph for exporting codes and codebooks.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        batch_size = word_vecs.size(0)
        vecs = []
        word_vecs_target = word_vecs
        loss = 0.0
        codebooks = []
        if self.adaptive:
            for i in range(len(self.sizes)):
                if self.codebooks_[i].size(1) == self.embd_dim:
                    codebooks.append(self.codebooks_[i].view(self.sizes[i],self.K,-1))
                else:

                    codebooks.append((self.codebooks_[i].mm(self.projs_[i])).view(self.sizes[i],self.K,-1))
            codebooks = torch.cat(codebooks,0).view(-1,self.embd_dim)
        else:
            codebooks = self.codebooks_
            
        for i in range(self.num_compressors):
            loss_,word_vecs_ = self.sub_compressors[i].compress(word_vecs,freq,group_mask,word_vecs_target,codebooks)
            vecs.append(word_vecs_[:,None,:])
        if self.top_word_basis is not None:
            group_mask_ = torch.ones(batch_size,self.top_word_basis.size(0)).cuda()
            loss_,word_vecs_ = self.sub_compressors[-1].compress(word_vecs,freq,group_mask_,word_vecs_target,None)
            vecs.append(word_vecs_[:,None,:])
                   
        # Decoding
        y_hat = torch.cat(vecs,1)
        y_hat = y_hat.sum(1)
        # Define loss
        if self.use_freq:
            loss_ = 0.5   * ((y_hat - word_vecs)**2).sum(dim=1)* (torch.log(freq) + 1e-6)
        else:
            loss_ = 0.5   * ((y_hat - word_vecs)**2).sum(dim=1)#* (torch.log(freq) + 1e-6)
        
        
        loss += loss_
        
        
        return loss, y_hat


    def compress_apply(self, word_vecs,group_mask):
        """
        Args:
            word_ids to be encoded
        """

        batch_size = word_vecs.size(0)
        vecs = []
        codes = []
        # Define codebooks
        num_codes = 0.0
        if self.adaptive:
            codebooks = []
            for i in range(len(self.sizes)):
                if self.codebooks_[i].size(1) == self.embd_dim:
                    codebooks.append(self.codebooks_[i].view(self.sizes[i],self.K,-1))
                else:

                    codebooks.append((self.codebooks_[i].mm(self.projs_[i])).view(self.sizes[i],self.K,-1))
            codebooks = torch.cat(codebooks,0).view(-1,self.embd_dim)
        else:
            codebooks = self.codebooks_
        for i in range(self.num_compressors):
            vecs_,_,codes_,num_code = self.sub_compressors[i].compress_apply(word_vecs,group_mask,codebooks)
            num_codes += num_code
            vecs.append(vecs_[:,None,:])
            codes.append(codes_[:,None,:].data.cpu())
        if self.top_word_basis is not None:
            group_mask_ = torch.ones(batch_size,self.top_word_basis.size(0)).cuda()
            vecs_,_,codes_,num_code = self.sub_compressors[-1].compress_apply(word_vecs,group_mask_,None)
            num_codes += num_code
            vecs.append(vecs_[:,None,:])
            codes.append(codes_[:,None,:].data.cpu())
        # Coding
        
        # Decoding
        reconstructed_embed = torch.cat(vecs,1)
        reconstructed_embed = reconstructed_embed.sum(1)
        
        return reconstructed_embed, word_vecs.cpu().data.numpy(),codes,num_codes
    def group(self):
            
        V = self.V
        if self.adaptive:
            word_lists = range(V)
            group_masks = np.zeros([V,self.M])
            num = 0
            for i in range(len(self.code_offsets)):

                lb  =self.cut_offs[i]
                if i == len(self.code_offsets) -1 :
                    rb  = V
                else:
                    rb = self.cut_offs[i+1]
                for j in range(lb,rb):
                    for k in range(self.M - self.code_offsets[i],self.M):

                        group_masks[j][k] = 1.0
                num += (rb -lb) * self.code_offsets[i]
            return group_masks,float(num)
        else:
            return np.ones([V,self.M]),self.V*self.M

    def print_compression_rate(self,quantization=0):
        _,num =self.group()
        compression_ = 0.0
        if quantization!=0:
            bits = quantization
        else:
            bits = 32
        if self.adaptive:
            for i in range(len(self.code_offsets)):
                if self.codebooks_[i].size(1) == self.embd_dim:
                    compression_ += self.codebooks_[i].size(0) * self.codebooks_[i].size(1) * bits
                else:

                    compression_ += (self.projs_[i].size(0) *self.projs_[i].size(1)\
                                     +self.codebooks_[i].size(0) * self.codebooks_[i].size(1)) * bits
            compression_ += len(self.code_offsets)*self.M
        else:
            
            compression_ += self.M*self.K*self.embd_dim * bits
        if self.mul:
            if self.use_full:
                compression_ +=  self.embd_dim * bits*self.num_compressors * self.M
            else:
                compression_ +=  self.embd_dim * bits*self.num_compressors 
            
        if self.top_word_basis is not None:
            compression_ += self.V * self.top_word_basis.size(0) *math.log(self.top_word_basis.size(1),2)
            
        compression_  += num *math.log(self.K,2) *self.num_compressors 
        
        print("")
        print("compression_rate:",compression_/(self.V * self.embd_dim * 32))
        return compression_/(self.V * self.embd_dim * 32)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


def init_weight_variable(shape,zero=False):
    
    sigma = 0.01
    initial = torch.zeros(shape)
    if not zero:
        initial.uniform_(-sigma, sigma)
    
    if torch.cuda.is_available():
        return nn.Parameter(initial.cuda())
    else:
        return nn.Parameter(initial)
    

    
class AdaptiveCompressor(nn.Module):

#     _TAU = 1.0
#     _BATCH_SIZE = 64

    def __init__(self, n_words, dim,n_codebooks, n_centroids,_TAU=1.0,lr=0.0001,codebooks=None,mul=True,test_in_epoch=True,use_full=False):
        
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        """
        
        super(AdaptiveCompressor,self).__init__()
        self.lr = lr
        self.M = n_codebooks
        self.K = n_centroids
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.V = n_words
        self.embd_dim = dim
        self.test_in_epoch = test_in_epoch
        self.use_full = use_full
#         self.bias =nn.Parameter(Variable(torch.rand([1,self.embd_dim])),requires_grad=True)
        
        self.linear_h = nn.Linear(self.embd_dim, self.M*self.K/2)
        self.linear_h.weight = init_weight_variable([self.M*self.K/2, self.embd_dim])
        
        self.linear_logit = nn.Linear(self.M*self.K/2,self.M *self.K)
        self.linear_logit.weight = init_weight_variable([self.M * self.K,self.M*self.K/2])
        self.codebooks = codebooks
            
        
        self._TAU = _TAU
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mul = mul
        if self.mul:
            if self.use_full:
                self.scale = nn.Parameter(Variable(torch.rand([self.M,1,self.embd_dim])),requires_grad=True)
            else:
                self.scale = nn.Parameter(Variable(torch.rand([1,1,self.embd_dim])),requires_grad=True)
         
     

    def init(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)
    def _gumbel_dist(self, shape, eps=1e-20):
        
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def _sample_gumbel_vectors(self, logits, temperature):
        
        y = logits + self._gumbel_dist(logits.size())
        return self.softmax( y / temperature)

    def _gumbel_softmax(self, logits, temperature, sampling=True):
        """
        Compute gumbel softmax.
        """
        return self._sample_gumbel_vectors(logits, temperature)


    def _st_gumbel_softmax(self,logits, temperature,sampling=True):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self._sample_gumbel_vectors(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
        
    def _encode(self, word_vecs):
        num_words,dim = word_vecs.size()
        M, K = self.M, self.K
    
        
        h = self.tanh(self.linear_h(word_vecs))
        
        logits = self.linear_logit(h)
        logits = torch.log(self.softplus(logits) + 1e-8)
        logits = logits.view([-1,    M, K])
    
        return  logits

    def _decode(self, gumbel_output,codebooks):
        if self.mul:
            codebooks =  self.tanh(self.scale.expand(self.M,self.K,self.embd_dim)).contiguous().view(self.M*self.K,-1) * codebooks
        return gumbel_output.mm(codebooks)
    

    
    def compress(self, word_vecs,freq,group_mask,target_word_vecs=None,codebooks=None):
        """Export the graph for exporting codes and codebooks.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        vocab_size,embed_size = self.V,self.embd_dim
        
        # Coding
        logits = self._encode(word_vecs)  # ~ (B, M, K)
#       
        batch_size = word_vecs.size(0)

        if codebooks is None:
            codebooks = self.codebooks
        # Discretization
        D = self._gumbel_softmax(logits, self._TAU, sampling=True)

        group_mask_ = group_mask[:,:,None].expand([batch_size,self.M,self.K]) 
        gumbel_output = D.view([-1,  self.M * self.K])  * group_mask_.contiguous().view(-1,self.M * self.K) # ~ (B, M * K)

        # Decoding
        y_hat = self._decode(gumbel_output,codebooks) 
        y_hat = y_hat.view(-1,embed_size) 

        # Define loss
        if target_word_vecs is not None:
            loss = 0.5   * ((y_hat - target_word_vecs)**2).sum(dim=1)* (torch.log(freq) + 1e-6)
        else:
            loss = 0.5   * ((y_hat - word_vecs)**2).sum(dim=1)* (torch.log(freq) + 1e-6)

        loss = loss
     
        
        return loss, y_hat 


    def compress_apply(self, word_vecs,group_mask,codebooks=None):
        """
        Args:
            word_ids to be encoded
        """

        batch_size = word_vecs.size(0)
        # Define codebooks
        if codebooks is None:
            codebooks = self.codebooks
        if self.mul:
            codebooks = self.tanh(self.scale).expand(self.M,self.K,self.embd_dim).contiguous().view(self.M*self.K,-1) * codebooks
        

        # Coding
        logits = self._encode(word_vecs)  # ~ (B, M, K)

        codes = logits.argmax(-1)
        
        codes_ = codes.view(-1,self.M)

        # Reconstruct
        offset = torch.arange(0,self.M).long().cuda() * self.K
        codes_with_offset = codes_ + offset.view(1, self.M)
        codes_with_offset = codes_with_offset.view(-1)
        
        selected_vectors = torch.index_select(codebooks,0,codes_with_offset)  # ~ (B, M, H)
        selected_vectors = selected_vectors.view(batch_size,self.M,-1) * group_mask[:,:,None]

        selected_vectors = selected_vectors.sum(1) 

        reconstructed_embed =   (selected_vectors)#+ self.bias#
        
        
        return reconstructed_embed, word_vecs.cpu().data.numpy(),codes_,group_mask.sum().data.cpu().numpy()

    def group(self):
            
        V = self.V
        word_lists = range(V)
        group_masks = np.ones([V,self.M])
        
        return group_masks,V *self.M

        
    def _train(self,embds, freq,sorted_index,batch_size=64,num_epochs=10,frozen_num=0):
        
        self.train()
        vocab_size = len(embds)
        word_list  = np.arange(0,vocab_size-frozen_num)
        group_masks,num =self.group()
        
        num_batches = (len(word_list)) / batch_size 
        best_loss = 1000.
        new_embedding = None
        if len(word_list) % batch_size:
            num_batches += 1
        self.print_compression_rate()
        for epoch in range(num_epochs):
            
            np.random.shuffle(word_list)
            _freq = [freq[w] for w in word_list]
            

            monitor_loss = 0.0
        
            for i in range(num_batches):

                words = word_list[i*batch_size:(i+1)*batch_size]
                freq_ = Variable(torch.Tensor(_freq[i*batch_size:(i+1)*batch_size])).cuda()
                word_vecs = Variable(torch.from_numpy(embds[words])).cuda()
                group_masks_ =  Variable(torch.from_numpy(group_masks[words])).float().cuda()
      
                loss,diff = self.compress(word_vecs,freq_,group_masks_)
                loss = loss.mean()
                loss.backward()
                
                if i %100 ==0 :
                    print monitor_loss / (i+1)
                
                torch.nn.utils.clip_grad_norm(self.parameters(),0.001)
                self.optimizer.step()

                self.zero_grad()

                monitor_loss += loss.data.cpu()
            monitor_loss /= num_batches
            monitor_loss = monitor_loss.data.cpu().numpy()
            print("epoch:",epoch,False,"loss:",monitor_loss)
            if monitor_loss <= best_loss * 0.99:
                best_loss = monitor_loss
                if self.test_in_epoch:
                    codes,new_embedding,compression_rate = self._test(embds,sorted_index,freq,batch_size)
                print("best_loss:", monitor_loss)
        if not self.test_in_epoch:
            codes,new_embedding,compression_rate = self._test(embds,sorted_index,freq,128)
        return codes,new_embedding,compression_rate
    
    def print_compression_rate(self,num,freq=None):
        if num is None:
            _,num =self.group( np.arange(0,self.V),int(self.M*(1-self.min_code_ratio)),freq)
        compression_ =  self.embd_dim * 32 + self.M*self.K*self.embd_dim * 32 +\
                    self.V * self.M*math.log(self.K,2)
        print("")
        print("compression_rate:",compression_/(self.V * self.embd_dim * 32))
        return compression_/(self.V * self.embd_dim * 32)
    def _test(self,embds, sorted_index,freq, batch_size=64):
        
        self.eval()
        vocab_size = self.V
        word_list  = np.arange(0,vocab_size)
        group_masks,num =self.group()
        
        new_embedding = np.zeros([self.V,self.embd_dim],dtype=np.float32)
        old_embedding = embds

        num_batches = (vocab_size) / batch_size 

        if vocab_size % batch_size:
            num_batches += 1
            
        distance = []
        reconstructed_ = []
        top_words = set()
        num_codes = 0
        codes_ = []
        for i in range(num_batches):

            words = word_list[i*batch_size:(i+1)*batch_size]

            word_vecs = Variable(torch.from_numpy(embds[words])).cuda()
            group_masks_ =  Variable(torch.from_numpy(group_masks[words])).float().cuda()

            reconstructed_vecs, original_vecs,codes,num_code_ = self.compress_apply(word_vecs,group_masks_)
            num_codes += num_code_
            reconstructed_vecs = reconstructed_vecs.cpu().data.numpy()
            reconstructed_.append(reconstructed_vecs)
            distance.append(np.linalg.norm(reconstructed_vecs - original_vecs,2,-1))
            codes_.append(codes)
        reconstruct = np.concatenate(reconstructed_,0)
        distance = np.concatenate(distance,-1)
        new_embedding[sorted_index] = reconstruct

        print("test:",np.linalg.norm(embds-new_embedding))#distance.mean()
        compression_rate = self.print_compression_rate()

        return codes_,new_embedding,compression_rate
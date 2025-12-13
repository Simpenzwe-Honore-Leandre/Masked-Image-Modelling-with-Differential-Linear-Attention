from torch.nn import functional as F, Linear, Module, Parameter, Sequential , Dropout , RMSNorm , GELU ,init 
from torch import exp , Tensor, tensor,  unsqueeze , sqrt 

from typing import Callable

class DIFFLINattn(Module):
  def __init__(self, config,
               lmbda: float =0.8,
               layer:int =1,
               kernel:Callable =F.elu):
    super().__init__()
    self.in_features = config.in_features if layer == 1 else config.out_features
    self.kernel = kernel
    self.out_features = config.out_features
    assert config.out_features > 0 and self.in_features > 0 and config.num_heads > 0 and layer > 0 , (
        f" none of: \n embed_dim={config.out_features},\n in_features={config.in_features} ,"
        f"\n num_heads={config.num_heads},\n layer={layer},\n must be less than zero"
        )
    self.num_heads = config.num_heads
    self.W_k = Linear( in_features=self.in_features, out_features=config.out_features,bias=config.bias)
    self.W_q = Linear( in_features=self.in_features, out_features=config.out_features,bias=config.bias)
    self.W_v = Linear( in_features=self.in_features, out_features=config.out_features,bias=config.bias)
    self.register_buffer("scale" , 1/ sqrt( tensor(config.out_features//2)) )
    self.lambda_params = Parameter( tensor(4 * [lmbda]) )
    self.lambda_init= 0.8 - 0.6 * exp( tensor(-0.3 * ( layer - 1)) )
    self.head_dim = config.out_features // config.num_heads
    assert self.head_dim * config.num_heads == config.out_features , (f" embedding dimension={config.out_features} must be divisible "
                                                     f"by number of heads={config.num_heads}" )
    self.norm_1 = RMSNorm(self.in_features,elementwise_affine=True)
    self.norm_2 = RMSNorm(self.head_dim,elementwise_affine=True)
    self.norm_3 = RMSNorm(config.out_features, elementwise_affine=True)
    self.ffn    = Sequential(
        Dropout(config.dropout),
        Linear(config.out_features, config.out_features,bias=config.bias),
        GELU(approximate=None),
        Dropout(config.dropout),
        Linear(config.out_features,config.out_features,bias=config.bias)
    )
    self.apply(self.init_weights)
  @staticmethod
  def init_weights(m:Module|Sequential)->None:
    if isinstance(m, (Linear)):
      init.orthogonal_(m.weight)

  def forward(self,x:Tensor,
              inverse:bool =False)->Tensor:
    assert len( x.shape ) >= 2, f"expected x with at least 2 dimensions"
    if len(x.shape) == 2:
      x = unsqueeze(x, 1)
    batch, seq_len = x.shape[0] , x.shape[1]
    x = self.norm_1(x)
    Q = self.W_q(x).view(batch, seq_len , self.num_heads, self.head_dim)
    K = self.W_k(x).view(batch, seq_len , self.num_heads, self.head_dim)
    V = self.W_v(x).view(batch, seq_len , self.num_heads, self.head_dim)

    Q_1,Q_2 = Q.chunk(2, dim=-1)
    K_1,K_2 = K.chunk(2, dim=-1)

    attn_w1 = self.kernel(Q_1) @ (self.kernel(K_1).transpose(-1,-2) * self.scale @ V)

    attn_w2 = self.kernel(Q_2) @ (self.kernel(K_2).transpose(-1,-2) * self.scale @ V)

    exps = exp( self.lambda_params[[0,2]] * self.lambda_params[[1,3]])
    self.lmbda = exps[0] - exps[1] + self.lambda_init

    attn = attn_w2 - self.lmbda * attn_w1 if inverse else attn_w1 - self.lmbda * attn_w2

    attn = (1 - self.lambda_init ) * self.norm_2( attn )

    attn = attn.contiguous().view(batch, seq_len, self.out_features)

    attn = self.norm_3( attn ) + attn

    attn = self.ffn( attn ) + attn

    return attn.squeeze()





class SDPA(Module):
  def __init__(self, config):
        super().__init__(self)
        self.num_heads = config.num_heads
        self.head_dim = config.out_features // config.num_heads
        assert self.head_dim * config.num_heads == config.out_features , \
          (f" embedding dimension={config.out_features} must be divisible "
           f"by number of heads={config.num_heads}" )
        self.out_features = config.out_features
        self.v_proj = Linear(config.in_features,config.out_features,bias=config.bias)
        self.k_proj = Linear(config.in_features,config.out_features,bias=config.bias)
        self.q_proj = Linear(config.in_features,config.out_features,bias=config.bias)
        self.ffn    = Sequential(Dropout(config.dropout),
                                Linear(config.out_features, config.out_features,bias=config.bias),
                                GELU(approximate=None),  
                                Dropout(config.dropout),
                                Linear(config.out_features,config.out_features,bias=config.bias))
        self.scale = config.out_features ** -0.5
        self.prenorm = RMSNorm(config.out_features)
        self.postnorm = RMSNorm(config.out_features)

        self.apply(self.init_weights)
  @staticmethod
  def init_weights(m:Module|Sequential)->None:
    if isinstance(m, (Linear)):
      init.orthogonal_(m.weight)

def forward(self,x:Tensor)->Tensor:
    assert len( x.shape ) >= 2, f"expected x with at least 2 dimensions"
    if len(x.shape) == 2:
      x = unsqueeze(x, 1)
    batch, seq_len = x.shape[0] , x.shape[1]
    x = self.prenorm(x)
    q = self.q_proj(x).view(batch,seq_len,self.num_heads, self.head_dim)
    k = self.k_proj(x).view(batch,seq_len,self.num_heads, self.head_dim)
    v = self.v_proj(x).view(batch,seq_len,self.num_heads, self.head_dim)
    attn = F.softmax(self.scale * q @ k.transpose(-1,-2)) @ v
    attn = attn.contiguous().view(batch,seq_len,self.out_features)
    attn = self.postnorm(attn) + attn
    attn = self.ffn(attn) + attn
    return attn
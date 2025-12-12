from torch.nn import functional as F, Conv2d, Linear, Module, Parameter, Sequential , Dropout , RMSNorm , GELU ,init 
from torch import exp , Tensor, tensor,  unsqueeze , sqrt 

from typing import Union,Optional,List,Tuple,Sequence, Callable

class DIFFLINattn(Module):
  def __init__(self, in_features:int,
               embed_dim: int,
               num_heads: int =16,
               lmbda: float =0.8,
               layer:int =1,
               kernel:Callable =F.elu,
               bias=False):
    super().__init__()
    assert embed_dim > 0 and in_features > 0 and num_heads > 0 and layer > 0 , (
        f" none of: \n embed_dim={embed_dim},\n in_features={in_features} ,"
        f"\n num_heads={num_heads},\n layer={layer},\n must be less than zero"
        )
    embed_dim = embed_dim + ( embed_dim % 2 )
    self.W_k = Linear( in_features=in_features, out_features=embed_dim,bias=bias)
    self.W_q = Linear( in_features=in_features, out_features=embed_dim,bias=bias)
    self.W_v = Linear( in_features=in_features, out_features=embed_dim,bias=bias)
    self.register_buffer("scale" , 1/ sqrt( tensor(embed_dim//2)) )
    self.embed_dim = embed_dim
    self.in_features = in_features
    self.lambda_params = Parameter( tensor(4 * [lmbda]) )
    self.lambda_init= 0.8 - 0.6 * exp( tensor(-0.3 * ( layer - 1)) )
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.kernel = kernel
    assert self.head_dim * num_heads == embed_dim , (f" embedding dimension={embed_dim} must be divisible "
                                                     f"by number of heads={num_heads}" )
    self.norm_1 = RMSNorm(in_features,elementwise_affine=True)
    self.norm_2 = RMSNorm(self.head_dim,elementwise_affine=True)
    self.norm_3 = RMSNorm(embed_dim, elementwise_affine=True)
    self.ffn    = Sequential(
        Dropout(),
        Linear(embed_dim, embed_dim,bias=bias),
        GELU(approximate=None),
        Dropout(),
        Linear(embed_dim,embed_dim,bias=bias)
    )
    self.apply(self.init_weights)
  @staticmethod
  def init_weights(m:Union[Module,Sequential])->None:
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

    attn = attn.contiguous().view(batch, seq_len, self.embed_dim)

    attn = self.norm_3( attn ) + attn

    attn = self.ffn( attn ) + attn

    return attn.squeeze()





class SDPA(Module):
  def __init__(self, in_features:int,
               embed_dim:int,
               num_heads:int,
               num_latent_tokens:int,
               bias:bool=False):
        super().__init__(self)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim , \
          (f" embedding dimension={embed_dim} must be divisible "
           f"by number of heads={num_heads}" )
        self.v_proj = Linear(in_features,embed_dim,bias=bias)
        self.k_proj = Linear(in_features,embed_dim,bias=bias)
        self.q_proj = Linear(in_features,embed_dim,bias=bias)
        self.num_latent_tokens = num_latent_tokens
        self.ffn    = Sequential(Dropout(),
                                Linear(embed_dim, embed_dim,bias=bias),
                                GELU(approximate=None),  
                                Dropout(),
                                Linear(embed_dim,embed_dim,bias=bias))
        self.scale = self.width ** -0.5
        self.prenorm = RMSNorm(embed_dim)
        self.postnorm = RMSNorm(embed_dim)

        self.apply(self.init_weights)
  @staticmethod
  def init_weights(m:Union[Module,Sequential])->None:
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
    attn = F.softmax(q @ k.transpose(-1,-2)) @ v
    attn = attn.contiguous().view(batch,seq_len,self.embed_dim)
    attn = self.postnorm(attn) + attn
    attn = self.ffn(attn)
    return attn
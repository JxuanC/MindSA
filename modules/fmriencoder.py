from modules.vit import ViT
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch

class GumbelSoftmaxSampler():
    """Sample based on a Gumbel-Max distribution.

    Use re-param trick for back-prop
    """
    def __init__(self, tau=1., device='cuda'):
        #self.num_samples = num_samples
        # self.num_points = num_points
        self.device = device
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0.),
                torch.tensor(1.))
        self.tau = tau

    def sampling(self, logits, num_samples, selected = None):
        if selected is None:
            gumbels = self.gumbel_dist.sample(logits.shape).to(logits)
            gumbels = (logits + gumbels)/self.tau
            y_soft = gumbels.softmax(-1)
            topk = torch.topk(gumbels, num_samples, dim=-1)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            pass

        return ret, y_soft#, topk.indices

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class fMRIViTEncoderConfig(PretrainedConfig):
    model_type = "fMRIViTEncoder"

    def __init__(
        self,
        fmri_dim, rois_len, embed_dim, depth, num_heads,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fmri_dim = fmri_dim
        self.rois_len = rois_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = embed_dim
        

class fMRIViTEncoder(PreTrainedModel):
    config_class = fMRIViTEncoderConfig
    def __init__(self, config):
        super(fMRIViTEncoder, self).__init__(config)
        #self.proj = nn.Linear(config.fmri_dim, 112 * 112 * 3)
        #self.patch_embed = PatchEmbed(112, 16, 3, config.embed_dim)
        self.mlp = SimpleMLP(config.embed_dim, 256, config.embed_dim)
        self.encoder = ViT(config.fmri_dim, config.rois_len, config.embed_dim, config.depth, config.num_heads)
        self.sampler = GumbelSoftmaxSampler()
        self.config = config

    def reshape_clip(self, clip_inputs):
        #class_token = clip_inputs[:,0,:]
        #patch_token = clip_inputs[:,1:,:]
        return self.mlp(clip_inputs)

    def forward(self, encoder_inputs, **kwargs):
        # encoder_inputs shape (batch, roi_num, roi_dim)    
        encoder_outputs = self.encoder(encoder_inputs)
        return encoder_outputs
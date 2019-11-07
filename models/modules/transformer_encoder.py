from torch.nn import Module,ModuleList
from torch.nn.modules.transformer import TransformerEncoder
from torch.nn.modules import Linear
import copy

class MyTransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None,in_size=768,d_model=128):
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.linear = Linear(in_size,d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = self.linear(src)

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

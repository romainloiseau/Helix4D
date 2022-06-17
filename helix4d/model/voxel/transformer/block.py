import torch

from ...sparseops import LayerNorm as TGLayerNorm
from .attention import Attention
from .utils import feed_forward


class TransformerBlockNOFFN(torch.jit.ScriptModule):

    def __init__(self, dim, transformer, *args, **kwargs):
        super().__init__()

        self.attention = Attention(
            dim, transformer)
        self.norm_attention = TGLayerNorm(dim)

        self.preln = transformer.architecture == "Pre-LN"
        self.postln = transformer.architecture == "Post-LN"

    @torch.jit.script_method
    def forward(self, x, batch, iq, ik, buckets, keystouse, valuestouse):
        x, proba, key, value = self.do_attention(
            x, batch, iq, ik, buckets, keystouse, valuestouse)

        return x, proba, key, value

    @torch.jit.script_method
    def do_attention(self, x, batch, iq, ik, buckets, keystouse, valuestouse):
        with torch.profiler.record_function("T"):
            out_attention, proba, key, value = self.attention(
                self.norm_attention(x, batch=batch) if self.preln else x, iq, ik, buckets, keystouse, valuestouse)

            with torch.profiler.record_function("+"):
                x = x + out_attention
            if self.postln:
                with torch.profiler.record_function("NORM"):
                    x = self.norm_attention(x, batch=batch)
            return x, proba, key, value

class TransformerBlockFFN(TransformerBlockNOFFN):

    def __init__(self, dim, transformer, *args, **kwargs):
        super().__init__()
        
        self.feed_forward = feed_forward(
            dim, drop=transformer.dropout,
            mul_feedforward=transformer.mul_ff
        )
        self.norm_feed_forward = TGLayerNorm(dim)

    @torch.jit.script_method
    def forward(self, x, batch, iq, ik, buckets, keystouse, valuestouse):
        x, proba, key, value = super().forward(x, batch, iq, ik, buckets, keystouse, valuestouse)
        x = self.do_feed_forward(x, batch)

        return x, proba, key, value

    @torch.jit.script_method
    def do_feed_forward(self, x, batch):
        with torch.profiler.record_function("FF"):
            with torch.profiler.record_function("ENC"):
                out_ff = self.feed_forward(self.norm_feed_forward(
                    x, batch=batch) if self.preln else x)

            with torch.profiler.record_function("+"):
                x = x + out_ff
            if self.postln:
                with torch.profiler.record_function("NORM"):
                    x = self.norm_feed_forward(x, batch=batch)
            return x

"""Read-only full-token global-attention image probe for Pi3."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image


class GlobalAttentionImageProbe:
    def __init__(self, model, output_dir, query_chunk=128):
        self.model, self.output_dir, self.query_chunk = model, Path(output_dir), query_chunk
        self.handles = []

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Pi3-large has 36 alternating decoder blocks: odd indices are global.
        for global_index, decoder_index in enumerate(range(1, len(self.model.decoder), 2)):
            self.handles.append(self.model.decoder[decoder_index].attn.register_forward_pre_hook(
                self._hook(global_index), with_kwargs=True))
        return self

    def __exit__(self, *unused):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer):
        def save(attn, args, kwargs):
            x, xpos = args[0], kwargs.get("xpos")
            b, n, c = x.shape
            qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
            if attn.rope is not None:
                q, k = attn.rope(q, xpos), attn.rope(k, xpos)
            output = np.memmap(self.output_dir / f"global_{layer:02d}_mean_heads.u16", mode="w+", dtype=np.uint16, shape=(n, n))
            kt = k.float().transpose(-2, -1)
            for start in range(0, n, self.query_chunk):
                end = min(start + self.query_chunk, n)
                prob = (torch.matmul(q[:, :, start:end].float(), kt) * attn.scale).softmax(-1).mean(1)[0]
                output[start:end] = (prob.clamp(0, 1).cpu().numpy() * 65535).round().astype(np.uint16)
            output.flush()
            Image.fromarray(output, mode="I;16").save(self.output_dir / f"global_{layer:02d}_mean_heads.tiff", compression="tiff_lzw")
        return save

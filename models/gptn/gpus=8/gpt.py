import torch.nn as nn
import torch.nn.functional as F
import torch

from .stage_first import StageFirst
from .stage_mid import StageMid
from .stage_last import StageLast

class GPTLanguageModel(nn.Module):

    def __init__(self, n_embd=384, n_head=6, block_size=256, vocab_size=1000, dropout=0.0, n_layer=8):
        super().__init__()
        assert(n_layer >= 8)
        n_layer_mid = (n_layer - 2) // 6
        n_layer_mid_last = n_layer - 2 - n_layer_mid * 5
        self.stage0 = StageFirst(n_embd, n_head, block_size, vocab_size, dropout, 1)
        self.stage1 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid)
        self.stage2 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid)
        self.stage3 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid)
        self.stage4 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid)
        self.stage5 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid)
        self.stage6 = StageMid(n_embd, n_head, block_size, dropout, n_layer_mid_last)
        self.stage7 = StageLast(n_embd, n_head, block_size, vocab_size, dropout, 1)

        self.block_size = block_size

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        out5 = self.stage5(out4)
        out6 = self.stage6(out5)
        out7 = self.stage7(out6)
        return out7

    def generate(self, idx, max_new_tokens=500, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def load_checkpoints(self, checkpoints, source):
        """Load checkpoints for each stage."""
        if source == 'pipedream':
            self.stage0.load_state_dict(torch.load(checkpoints['stage0'], weights_only=True)['state_dict']['module0'])
            self.stage1.load_state_dict(torch.load(checkpoints['stage1'], weights_only=True)['state_dict']['module0'])
            self.stage2.load_state_dict(torch.load(checkpoints['stage2'], weights_only=True)['state_dict']['module0'])
            self.stage3.load_state_dict(torch.load(checkpoints['stage3'], weights_only=True)['state_dict']['module0'])
            self.stage4.load_state_dict(torch.load(checkpoints['stage4'], weights_only=True)['state_dict']['module0'])
            self.stage5.load_state_dict(torch.load(checkpoints['stage5'], weights_only=True)['state_dict']['module0'])
            self.stage6.load_state_dict(torch.load(checkpoints['stage6'], weights_only=True)['state_dict']['module0'])
            self.stage7.load_state_dict(torch.load(checkpoints['stage7'], weights_only=True)['state_dict']['module0'])
        else:
            self.stage0.load_state_dict(torch.load(checkpoints['stage0'], weights_only=True)['state_dict'])
            self.stage1.load_state_dict(torch.load(checkpoints['stage1'], weights_only=True)['state_dict'])
            self.stage2.load_state_dict(torch.load(checkpoints['stage2'], weights_only=True)['state_dict'])
            self.stage3.load_state_dict(torch.load(checkpoints['stage3'], weights_only=True)['state_dict'])
            self.stage4.load_state_dict(torch.load(checkpoints['stage4'], weights_only=True)['state_dict'])
            self.stage5.load_state_dict(torch.load(checkpoints['stage5'], weights_only=True)['state_dict'])
            self.stage6.load_state_dict(torch.load(checkpoints['stage6'], weights_only=True)['state_dict'])
            self.stage7.load_state_dict(torch.load(checkpoints['stage7'], weights_only=True)['state_dict'])


    
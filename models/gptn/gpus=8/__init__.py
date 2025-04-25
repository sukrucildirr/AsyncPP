from .gpt import GPTLanguageModel
from .stage_first import StageFirst
from .stage_mid import StageMid
from .stage_last import StageLast

def arch():
    return "gptn"

def model(criterion, vocab_size, block_size, dropout=0.0, n_layer=8, n_head=6, n_embd=384):
    assert(n_layer >= 8)
    n_layer_mid = (n_layer - 2) // 6
    n_layer_mid_last = n_layer - 2 - n_layer_mid * 5
    return [
        (StageFirst(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=1), ["input0"], ["out0"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), ["out0"], ["out1"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), ["out1"], ["out2"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), ["out2"], ["out3"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), ["out3"], ["out4"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid), ["out4"], ["out5"]),
        (StageMid(block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=n_layer_mid_last), ["out5"], ["out6"]),
        (StageLast(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_head=n_head, n_embd=n_embd, n_layer=1), ["out6"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]

def full_model(vocab_size, block_size, dropout=0.0, n_layer=6, n_head=6, n_embd=384):
    return GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, dropout=dropout, n_layer=n_layer, n_head=n_head, n_embd=n_embd)

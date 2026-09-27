import json
import os
import random
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, "vendor_cuda")
sys.path.insert(1, "vendor")
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from paradigm_model import clean_train, score


SEED = 20260925
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("device", DEVICE, flush=True)

PAD, BOS, EOS, UNK = 0, 1, 2, 3
RESERVED = {"<PAD>": PAD, "<BOS>": BOS, "<EOS>": EOS, "<UNK>": UNK}


def add_frequency(train, other):
    train = train.copy()
    other = other.copy()
    train["lemma_frequency"] = pd.to_numeric(train.get("lemma_frequency", 0), errors="coerce").fillna(0)
    if "lemma_frequency" not in other:
        other["lemma_frequency"] = 0.0
    other["lemma_frequency"] = pd.to_numeric(other.lemma_frequency, errors="coerce").fillna(0)
    if "yat_flag" not in other:
        other["yat_flag"] = 0
    return train, other


def input_tokens(row, freq_edges):
    lemma = str(row.lemma_ru)
    f = max(0.0, float(getattr(row, "lemma_frequency", 0) or 0))
    bucket = int(np.digitize(np.log1p(f), freq_edges))
    tokens = list(lemma)
    tokens.extend(["<SEP>", f"<POS_{row.pos}>"])
    tokens.extend(f"<M_{t}>" for t in str(row.feats).split(";") if t)
    tokens.extend([f"<DIA_{row.dialect}>", f"<YAT_{int(row.yat_flag)}>", f"<FREQ_{bucket}>"])
    return tokens


def build_examples(tr, sources, freq_edges):
    inps, outs = [], []
    for row in tr.itertuples(index=False):
        inps.append(input_tokens(row, freq_edges))
        outs.append(list(str(row.form_vvz)))
    for row in sources.itertuples(index=False):
        inps.append(input_tokens(row, freq_edges))
        outs.append(list(str(row.form_vvz)))
    return inps, outs


def create_vocab(inps, outs):
    vocab = dict(RESERVED)
    for seq in inps + outs:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(seqs, vocab):
    return [torch.tensor([vocab.get(t, UNK) for t in s], dtype=torch.long) for s in seqs]


class AttentiveGRU(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.init = nn.Linear(hidden * 2, hidden * 2)
        self.query = nn.Linear(hidden * 2, hidden * 2, bias=False)
        self.cell = nn.GRUCell(emb + hidden * 2, hidden * 2)
        self.out = nn.Linear(hidden * 4, vocab_size)

    def encode_src(self, x):
        mask = x.ne(PAD)
        z, h = self.encoder(self.emb(x))
        state = torch.tanh(self.init(torch.cat([h[-2], h[-1]], dim=-1)))
        return z, mask, state

    def step(self, token, state, enc, mask):
        scores = torch.bmm(enc, self.query(state).unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc).squeeze(1)
        state = self.cell(torch.cat([self.emb(token), ctx], dim=-1), state)
        logits = self.out(torch.cat([state, ctx], dim=-1))
        return logits, state

    def forward(self, x, y):
        enc, mask, state = self.encode_src(x)
        steps = y.shape[1]
        inp = torch.full((x.shape[0],), BOS, dtype=torch.long, device=x.device)
        outputs = []
        for t in range(steps):
            logits, state = self.step(inp, state, enc, mask)
            outputs.append(logits)
            inp = y[:, t]
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, x, max_len=56):
        enc, mask, state = self.encode_src(x)
        inp = torch.full((x.shape[0],), BOS, dtype=torch.long, device=x.device)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        out = []
        for _ in range(max_len):
            logits, state = self.step(inp, state, enc, mask)
            inp = logits.argmax(-1)
            inp = torch.where(done, torch.full_like(inp, EOS), inp)
            out.append(inp)
            done |= inp.eq(EOS)
            if bool(done.all()):
                break
        return torch.stack(out, dim=1)


def batches(indices, batch_size, inps, outs, shuffle):
    ix = np.asarray(indices)
    if shuffle:
        np.random.shuffle(ix)
        # Sort short windows by input length so batches use less padding.
        windows = [ix[i : i + batch_size * 24] for i in range(0, len(ix), batch_size * 24)]
        batch_ix = []
        for w in windows:
            w = sorted(w, key=lambda i: len(inps[i]))
            batch_ix.extend([w[j : j + batch_size] for j in range(0, len(w), batch_size)])
        random.shuffle(batch_ix)
    else:
        batch_ix = [ix[i : i + batch_size] for i in range(0, len(ix), batch_size)]
    for ids in batch_ix:
        x = pad_sequence([inps[int(i)] for i in ids], batch_first=True, padding_value=PAD)
        y = pad_sequence([outs[int(i)] for i in ids], batch_first=True, padding_value=PAD)
        yield x.to(DEVICE), y.to(DEVICE)


def main():
    train = pd.read_csv("data/train.csv").fillna("")
    dev = pd.read_csv("data/dev.csv").fillna("")
    test = pd.read_csv("data/test_features.csv").fillna("")
    seeds = pd.read_csv("data/wug_seeds.csv").fillna("")
    train, _ = add_frequency(train, train)
    _, dev = add_frequency(train, dev)
    _, test = add_frequency(train, test)
    # Make seed examples carry the frequency feature visible for that same test lemma.
    freq_map = test.drop_duplicates("lemma_ru").set_index("lemma_ru").lemma_frequency.to_dict()
    seeds["lemma_frequency"] = seeds.lemma_ru.map(freq_map).fillna(0.0)
    if "yat_flag" not in seeds:
        seeds["yat_flag"] = 0
    clean, similarities = clean_train(train, threshold=0.25)
    print("train rows", len(train), "kept", len(clean), "discarded obvious noise", int((similarities <= .25).sum()), flush=True)
    logfreq = np.log1p(pd.to_numeric(clean.lemma_frequency, errors="coerce").fillna(0).values.astype(float))
    freq_edges = np.quantile(logfreq, np.linspace(.125, .875, 7))
    # The provided wug labels are training supervision for unseen lemmas.
    inps, chars = build_examples(clean, seeds, freq_edges)
    # Output characters come only from labeled train/seed rows; input symbols include all visible test/dev forms.
    extra_in = []
    for frame in [dev, test]:
        extra_in.extend(input_tokens(r, freq_edges) for r in frame.itertuples(index=False))
    vocab = create_vocab(inps + extra_in, chars)
    ivocab = {v: k for k, v in vocab.items()}
    xtrain = encode(inps, vocab)
    ytrain = encode([s + ["<EOS>"] for s in chars], vocab)
    xdev = encode([input_tokens(r, freq_edges) for r in dev.itertuples(index=False)], vocab)
    ydev = encode([list(str(v)) + ["<EOS>"] for v in dev.form_vvz], vocab)
    model = AttentiveGRU(len(vocab), emb=128, hidden=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=1e-5)
    lossfn = nn.CrossEntropyLoss(ignore_index=PAD)
    batch_size = 256 if DEVICE.type == "cuda" else 128
    epochs = int(os.environ.get("MORPH_EPOCHS", "8"))
    best = -1
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in batches(range(len(xtrain)), batch_size, xtrain, ytrain, True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, y)
            loss = lossfn(logits.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        model.eval()
        pred = []
        for x, _ in batches(range(len(xdev)), 512, xdev, ydev, False):
            ids = model.generate(x).cpu().tolist()
            for seq in ids:
                chars_out = []
                for i in seq:
                    if i in (EOS, PAD):
                        break
                    chars_out.append(ivocab.get(i, ""))
                pred.append("".join(chars_out))
        val = score(dev.form_vvz.astype(str).values, pred)
        print("epoch", epoch, "loss", round(total_loss / max(1, n), 4), "dev", val, flush=True)
        if val["score"] > best:
            best = val["score"]
            torch.save(model.state_dict(), "char_model.pt")
            with open("char_vocab.json", "w", encoding="utf-8") as f:
                json.dump({"vocab": vocab, "freq_edges": [float(x) for x in freq_edges]}, f, ensure_ascii=False)
            pd.DataFrame({"id": dev.id, "form_vvz": pred}).to_csv("dev_char_predictions.csv", index=False)
        if epoch >= 3 and val["score"] < best - .015:
            break
    # Reload the best checkpoint and decode test rows, using the test-only vocabulary metadata.
    with open("char_vocab.json", encoding="utf-8") as f:
        payload = json.load(f)
    vocab = payload["vocab"]
    model = AttentiveGRU(len(vocab), emb=128, hidden=128).to(DEVICE)
    model.load_state_dict(torch.load("char_model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    xtest = encode([input_tokens(r, freq_edges) for r in test.itertuples(index=False)], vocab)
    dummy = [torch.tensor([EOS]) for _ in range(len(xtest))]
    out = []
    for x, _ in batches(range(len(xtest)), 512, xtest, dummy, False):
        ids = model.generate(x).cpu().tolist()
        for seq in ids:
            s = []
            for i in seq:
                if i in (EOS, PAD):
                    break
                s.append(ivocab.get(i, ""))
            out.append("".join(s))
    pd.DataFrame({"id": test.id, "form_vvz": out}).to_csv("test_char_predictions.csv", index=False)
    print("saved test predictions", len(out), flush=True)


if __name__ == "__main__":
    main()

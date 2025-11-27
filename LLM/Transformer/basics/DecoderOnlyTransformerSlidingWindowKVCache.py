# ----------------------------
# Rotary Position Embedding
# ----------------------------

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("n,d->nd", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self):
        return self.cos, self.sin

# ----------------------------
# Attention with KV Cache
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kv_cache_window: Optional[int] = None,  # e.g., 256 for sliding window
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_cache_window = kv_cache_window

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # print("x size", x.shape)
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        cos, sin = self.rotary_emb() 
        # print("position_ids", position_ids.shape)
        if position_ids is not None:
            cos = cos.squeeze(0).squeeze(0)[position_ids].unsqueeze(0).unsqueeze(0)
            sin = sin.squeeze(0).squeeze(0)[position_ids].unsqueeze(0).unsqueeze(0)
        # print("q", q.shape)
        # print("cos", cos.shape)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if use_cache:
            if past_kv is not None:
                past_k, past_v = past_kv
                # print("past_k", past_k.shape)
                # print("k", k.shape)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            # KV cache eviction: sliding window
            if self.kv_cache_window is not None and k.size(2) > self.kv_cache_window:
                k = k[:, :, -self.kv_cache_window:]
                v = v[:, :, -self.kv_cache_window:]

            present_kv = (k, v)
        else:
            present_kv = None

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hd) -> (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return y, present_kv

# ----------------------------
# Transformer Block
# ----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, kv_cache_window=None):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, kv_cache_window)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, use_cache=False, past_kv=None, position_ids=None):
        attn_out, present_kv = self.attn(self.ln1(x), use_cache, past_kv, position_ids)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv

# ----------------------------
# Decoder-Only Transformer
# ----------------------------

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        kv_cache_window: Optional[int] = None,  # e.g., 256
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.kv_cache_window = kv_cache_window

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout, kv_cache_window)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # self.tok_emb.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Input length {T} exceeds max_seq_len {self.max_seq_len}"

        x = self.tok_emb(input_ids)  # (B, T, embed_dim)
        x = self.dropout(x)

        presents = []
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        for block, past_kv in zip(self.blocks, past_key_values):
            x, present_kv = block(x, use_cache=use_cache, past_kv=past_kv, position_ids=position_ids)
            # if present_kv is not None:
            #     print("present_kv[0]", present_kv[0].shape)
            presents.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": presents if use_cache else None,
        }

# ----------------------------
# Training
# ----------------------------

def train_model(model, optimizer, input_ids, labels, N_epochs = 101):
    model.train()
    for epoch in range(N_epochs):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels,
                        use_cache=False, past_key_values=None,
                        position_ids = torch.arange(0, input_ids.shape[-1]))
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Train loss: {loss:.4f}")
    return loss.item()

# ----------------------------
# Generation with KV Cache & Eviction
# ----------------------------

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    model.eval()
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Batched generation with cache not implemented here"

    generated = input_ids.clone()
    past_key_values = None
    position_ids = torch.arange(generated.shape[1], device=input_ids.device)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated if past_key_values is None else generated[:, -1:],
            use_cache=True,
            past_key_values=past_key_values,
            position_ids=position_ids if past_key_values is None else position_ids[-1:],
        )
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"] # KV cache here

        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer_eos_token_id:  # Define your EOS token ID
            break

        generated = torch.cat([generated, next_token], dim=1)
        position_ids = torch.cat([position_ids, position_ids[-1:] + 1], dim=-1)
        # print("position_ids", position_ids)

    return generated

# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Dummy config
    vocab_size = 10000
    tokenizer_eos_token_id = 0  # <-- set appropriately

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=3,
        num_heads=4,
        ffn_dim=1024,
        max_seq_len=512,
        kv_cache_window=256,  # enables sliding window attention
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Dummy data for training
    input_ids = torch.randint(1, vocab_size, (7, 128))
    # print("input_ids", input_ids.shape)
    labels = input_ids.clone()
    loss = train_model(model, optimizer, input_ids, labels)
    print()

    # Generate
    prompt = torch.randint(1, vocab_size, (1, 10))
    output = generate(model, prompt, max_new_tokens=20, temperature=0.9, top_k=50)
    print("Generated tokens:", output.squeeze().tolist())

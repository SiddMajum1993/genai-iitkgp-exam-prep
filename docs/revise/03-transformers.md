# Lecture 3 — Transformer Architecture, Attention, Tokenization & Decoding
**Instructor:** Prof. Niloy Ganguly

## TL;DR (5 bullets)
- Self-attention eliminates sequential processing: every token attends to every other token in O(1) path length, enabling parallelism and long-range dependency capture that RNNs could not achieve.
- Q/K/V projection matrices are the core learnable parameters: Query (what I seek), Key (what I offer), Value (what I give). Attention = softmax(QK^T / √d_k) · V.
- Multi-head attention captures multiple relationship types in parallel (e.g., syntax, coreference, semantics, position) by running h independent Q/K/V projections.
- Positional encoding injects order via sinusoidal functions added to embeddings: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)). Attention itself is permutation-invariant.
- Three attention patterns define modern transformers: self-attention (encoder; Q=K=V from same sequence), masked self-attention (decoder; causal mask prevents future leakage), cross-attention (decoder reads encoder; Q from decoder, K/V from encoder).

## Exam-relevant concepts

### Pre-Attention RNN/Seq2Seq Limitations
- **Definition:** Seq2seq with RNN encoders compressed entire input into single fixed-size context vector; gradients vanished/exploded; sequential processing prevented parallelism; uniform treatment of input tokens.
- **When it matters (exam trap):** Questions contrasting RNN vs Transformer bottlenecks. RNNs have O(n) dependency path (information travels through n steps), Transformers have O(1) (any two tokens connect directly in one attention step).
- **Common confusion:** "Attention layers" in RNN-based seq2seq only ran decoder→encoder, not encoder→encoder. That's NOT self-attention.

### Self-Attention Mechanism
- **Definition:** Each token attends to all other tokens in the same sequence. Q, K, V all derived from the same input X.
- **Formula:** Attention(Q, K, V) = softmax(QK^T / √d_k) · V
- **When it matters (exam trap):** Self-attention is permutation-equivariant (shuffle tokens → same attention weights if positional encoding is not added). Order must come from positional encoding.
- **Common confusion:** Attention score = QK^T (dot product). Scaling by √d_k prevents gradient vanishing. Without scaling, softmax saturates for large d_k.

### Query, Key, Value (Q/K/V)
- **Definition:** 
  - Query (Q): what the current token is looking for.
  - Key (K): what each token advertises it contains.
  - Value (V): the actual information a token contributes.
- **Formula:** Q = X · W_Q, K = X · W_K, V = X · W_V (each W is d×d projection matrix).
- **When it matters (exam trap):** In self-attention Q=K=V come from same X. In cross-attention: Q from decoder, K/V from encoder. Confusing the source is a common MCQ trick.
- **Common confusion:** "Why three matrices?" Because they learn different roles: Q-K matching (relevance) and V-aggregation (content blending) are separate operations.

### Scaled Dot-Product Attention
- **Formula:** Score_ij = (q_i · k_j) / √d_k, then softmax(Scores) → attention weights α, Output = α · V.
- **Numbers:** Scaling factor = √d_k. For d_k=64 (common head dim), √64 = 8.
- **When it matters (exam trap):** Without √d_k, dot products grow large → softmax saturates → gradients vanish. This is Training Trick #3.
- **Common confusion:** d_k is head dimension (d_model / n_heads), NOT d_model. For d_model=512, n_heads=8 → d_k=64.

### Multi-Head Attention
- **Definition:** Run h independent attention heads in parallel, each with own W_Q, W_K, W_V projections. Concatenate outputs and project with W_O.
- **Formula:** MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O, where head_i = Attention(Q·W_Q^(i), K·W_K^(i), V·W_V^(i)).
- **Numbers:** GPT-2 small: 12 heads. BERT-Base: 12 heads. Head dimension = d_model / n_heads = 512 / 8 = 64.
- **When it matters (exam trap):** Different heads learn different patterns (syntax, coreference, semantics, position). A single-head model cannot capture multiple relationship types simultaneously.
- **Common confusion:** "Why concatenate then project?" Because each head outputs d_k dims; concatenating h heads gives h·d_k = d_model dims; W_O projects back to d_model.

### Positional Encoding
- **Definition:** Sinusoidal position vectors added to token embeddings to inject sequence order. Attention itself has no notion of position.
- **Formula:** 
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  - Final input: x_i = token_embedding(i) + PE(pos_i)
- **When it matters (exam trap):** Without PE, "The cat chased the mouse" and "The mouse chased the cat" are identical to the model (permutation-invariant). PE is NOT learned; it's deterministic.
- **Common confusion:** "Why sin/cos?" Because sin(pos+k) and cos(pos+k) are linear combinations of sin(pos) and cos(pos) → model can learn relative positions. Also generalizes beyond training sequence lengths.

### Residual Connections
- **Definition:** Add input of a sub-layer to its output: x_out = x_in + SubLayer(x_in). Prevents gradient vanishing in deep networks.
- **When it matters (exam trap):** Training Trick #1. Every self-attention and FFN block is wrapped: x = x + Attention(x), then x = x + FFN(x).
- **Common confusion:** Residual connections are NOT the same as skip connections in CNNs (though conceptually similar). They stabilize training by allowing gradients to flow directly backward.

### Layer Normalization
- **Definition:** Normalize activations across the feature dimension (not batch dimension) before each sub-layer. μ = mean(x), σ = std(x), x_norm = (x - μ) / σ, then scale/shift with learned γ, β.
- **When it matters (exam trap):** Training Trick #2. LayerNorm is applied AFTER residual addition in original Transformer (Post-LN), but modern models often use Pre-LN (norm before sub-layer).
- **Common confusion:** LayerNorm normalizes per token (across d_model dims). BatchNorm normalizes per feature (across batch). Transformers use LayerNorm because batch size may vary during inference.

### Masked Self-Attention (Causal Mask)
- **Definition:** In decoder, prevent token at position i from attending to positions j > i (future tokens). Set scores_ij = -∞ for j > i before softmax.
- **When it matters (exam trap):** Mask is triangular (lower-left = allowed, upper-right = blocked). During training, entire target sequence is processed in parallel with mask. During inference, generation is autoregressive (one token per step).
- **Common confusion:** "Why mask during training?" Because the full target sequence is available during training (teacher forcing). Without masking, the model would cheat by seeing future tokens.

### Cross-Attention (Encoder-Decoder Attention)
- **Definition:** Decoder attends to encoder outputs. Q from decoder, K and V from encoder. Allows decoder to read source context dynamically.
- **Formula:** CrossAttn(Q_dec, K_enc, V_enc) = softmax(Q_dec · K_enc^T / √d_k) · V_enc.
- **When it matters (exam trap):** Q from one sequence, K/V from another → bridges two sequences. Self-attention cannot do this (all from same sequence).
- **Common confusion:** In translation (EN→FR), encoder processes "The cat sat" once. Decoder generates "Le chat" token-by-token, cross-attending to encoder states at each step.

### Autoregressive Generation
- **Definition:** Generate one token at a time. Start with &lt;BOS&gt;, predict next token, append to input, repeat until &lt;EOS&gt; or max_length.
- **When it matters (exam trap):** Greedy decoding = argmax at each step (deterministic). Beam search = keep top-k candidates. Sampling = introduce randomness (temperature, top-k, top-p).
- **Common confusion:** During training, decoder sees entire target sequence (with mask). During inference, decoder generates incrementally. The mask grows from 1×1 to 2×2 to 3×3 as sequence lengthens.

### Feed-Forward Network (FFN)
- **Definition:** Two-layer MLP applied independently to each token. d_ff = 4 × d_model (e.g., 512 → 2048 → 512).
- **Formula:** FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2 (ReLU activation).
- **When it matters (exam trap):** FFN is position-wise (no interaction between tokens). Attention mixes tokens; FFN processes each token independently.
- **Common confusion:** "Why 4×?" Empirical finding from "Attention Is All You Need" paper. Larger intermediate dimension gives model more capacity.

### Transformer Encoder vs Decoder
- **Definition:**
  - Encoder: Self-attention + FFN, repeated N times. Bidirectional (sees full input). Used in BERT.
  - Decoder: Masked self-attention + cross-attention + FFN, repeated N times. Causal (sees only past). Used in GPT (decoder-only, no cross-attention).
- **When it matters (exam trap):** Encoder-only (BERT) for classification/NER. Decoder-only (GPT) for generation. Encoder-decoder (T5/BART) for translation/summarization.
- **Common confusion:** GPT is "decoder-only" but does NOT use cross-attention (no encoder). It only uses masked self-attention.

## Diagrams / architectures / API sketches described

### Transformer Encoder Block
```
Input Embedding + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓ (+ residual)
Layer Norm
    ↓
Feed-Forward Network (d_model → d_ff → d_model)
    ↓ (+ residual)
Layer Norm
    ↓
Output (repeat 6×)
```

### Transformer Decoder Block
```
Output Embedding + Positional Encoding
    ↓
Masked Multi-Head Self-Attention (causal)
    ↓ (+ residual)
Layer Norm
    ↓
Multi-Head Cross-Attention (Q from decoder, K/V from encoder)
    ↓ (+ residual)
Layer Norm
    ↓
Feed-Forward Network
    ↓ (+ residual)
Layer Norm
    ↓
Linear + Softmax → Token Probabilities
(repeat 6×, then autoregressive loop)
```

### Attention Computation Steps
1. **Input:** X ∈ ℝ^(n×d), where n = sequence length, d = embedding dim.
2. **Projection:** Q = X·W_Q, K = X·W_K, V = X·W_V (each W is d×d).
3. **Scores:** Scores = Q·K^T (shape: n×n). Each entry score_ij = q_i · k_j.
4. **Scale:** Scores / √d_k (prevent saturation).
5. **Softmax:** A = softmax(Scores) (row-wise; each row sums to 1).
6. **Output:** O = A · V (weighted sum of values).

### Multi-Head Split and Concat
- **Input:** Q, K, V ∈ ℝ^(n×d_model).
- **Split:** Reshape into h heads: Q → (n, h, d_k), where d_k = d_model / h.
- **Parallel Attention:** Each head computes Attention(Q_i, K_i, V_i) → O_i ∈ ℝ^(n×d_k).
- **Concat:** Concatenate h outputs → (n, d_model).
- **Project:** Multiply by W_O (d_model × d_model) → final output.

### Positional Encoding Matrix
- **Shape:** PE ∈ ℝ^(max_len × d_model).
- **Example (d_model=512, max_len=512):**
  - pos=0: [sin(0/10000^0), cos(0/10000^0), sin(0/10000^(2/512)), ...]
  - pos=1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/512)), ...]
- **Usage:** x_final = x_token + PE[pos].

### Causal Mask (Autoregressive)
```
       j=0  j=1  j=2  j=3  (Key positions)
i=0    ✓    ✕    ✕    ✕    (Query pos 0 sees only itself)
i=1    ✓    ✓    ✕    ✕    (Query pos 1 sees 0,1)
i=2    ✓    ✓    ✓    ✕    (Query pos 2 sees 0,1,2)
i=3    ✓    ✓    ✓    ✓    (Query pos 3 sees 0,1,2,3)

✓ = score computed normally
✕ = score set to -∞ before softmax → weight ≈ 0
```

### PyTorch Multi-Head Attention Sketch
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads  # 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V):
        B = Q.size(0)
        # Project and split into heads
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1,2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1,2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1,2)
        # Scaled dot-product attention per head
        scores = Q @ K.transpose(-2,-1) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        # Concat heads and project
        out = out.transpose(1,2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)
```

## Likely MCQ angles

1. **Self-attention vs Cross-attention source confusion:** Q/K/V all from same sequence (self) vs Q from decoder, K/V from encoder (cross). Expect a scenario: "In encoder-decoder translation, which attention layer allows the French decoder to read English input?" → Cross-attention.

2. **Positional encoding necessity:** "What happens if you remove positional encoding?" → Model becomes permutation-invariant, cannot distinguish word order. "Why sinusoidal?" → Deterministic, generalizes beyond training length, enables relative position learning.

3. **Masked self-attention during training vs inference:** "Why does the decoder use a causal mask during training?" → Prevents future token leakage even though full target is available (teacher forcing). "During inference?" → Mask grows incrementally (1×1, 2×2, 3×3).

4. **Scaling factor √d_k:** "Why divide QK^T by √d_k?" → Prevents softmax saturation for large d_k. "What is d_k for d_model=768, n_heads=12?" → 768/12 = 64.

5. **Residual connections and LayerNorm:** "What is Training Trick #1/#2?" → Residual = x + SubLayer(x), LayerNorm = normalize per token across features. "Why residual?" → Gradient flow in deep networks.

6. **Encoder-only vs Decoder-only vs Encoder-Decoder:** "Which architecture for classification?" → BERT (encoder-only). "Which for text generation?" → GPT (decoder-only). "Which for translation?" → T5/BART (encoder-decoder).

## One-line flashcards

- Q: Attention formula → A: softmax(QK^T / √d_k) · V
- Q: Q/K/V source in self-attention → A: All from same input X
- Q: Q/K/V source in cross-attention → A: Q from decoder, K/V from encoder
- Q: Why scale by √d_k → A: Prevent softmax saturation, avoid vanishing gradients
- Q: d_k for d_model=512, n_heads=8 → A: 512/8 = 64
- Q: GPT-2 small parameter count → A: ~117M (12 layers, 12 heads, d_model=768)
- Q: Positional encoding formula (even dims) → A: PE(pos, 2i) = sin(pos / 10000^(2i/d))
- Q: Positional encoding formula (odd dims) → A: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- Q: Why positional encoding → A: Attention is permutation-invariant; PE injects order
- Q: Multi-head attention output dim → A: Concat(h heads × d_k) · W_O → d_model
- Q: FFN expansion factor → A: d_ff = 4 × d_model (e.g., 512 → 2048 → 512)
- Q: Residual connection formula → A: x_out = x_in + SubLayer(x_in)
- Q: Causal mask shape for seq_len=4 → A: Lower-triangular 4×4 (✓ below diagonal, ✕ above)
- Q: Autoregressive generation stop condition → A: &lt;EOS&gt; token or max_length reached
- Q: Encoder architecture (BERT) → A: Self-attention + FFN, bidirectional, no cross-attention
- Q: Decoder architecture (GPT) → A: Masked self-attention + FFN, causal, no cross-attention
- Q: Encoder-decoder architecture (T5) → A: Encoder (self-attn) + Decoder (masked self-attn + cross-attn)
- Q: RNN dependency path length → A: O(n) (sequential)
- Q: Transformer dependency path length → A: O(1) (direct attention)
- Q: LayerNorm vs BatchNorm → A: LayerNorm per token (across features), BatchNorm per feature (across batch)
- Q: Attention as hashtable analogy → A: Query matches keys (fuzzy), returns weighted sum of values
- Q: Head 1 learns coreference, Head 2 learns syntax → A: Multi-head captures multiple relationship types
- Q: Why Transformers enable pretraining → A: Q/K/V learn general language patterns, transfer to any task
- Q: BERT pretraining objective → A: Masked LM (15% tokens) + Next Sentence Prediction
- Q: GPT pretraining objective → A: Causal LM (predict next token)

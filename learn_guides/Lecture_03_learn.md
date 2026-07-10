# Lecture 3 — Transformer Architecture, Attention, Tokenization, and Decoding
**Estimated read time:** 25–30 min · **Difficulty:** Foundation

## What this lecture is about

This lecture introduces the Transformer, the architecture that powers every modern LLM from BERT to GPT-4 to Claude. If you have used ChatGPT but do not know what happens inside when you hit Send, this is where that fog clears. You will learn why the world moved away from RNNs, how attention works at the matrix level, what Q/K/V actually mean, why positional encodings exist, and how a decoder generates text one token at a time. By the end, you will understand the difference between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) models, and when to pick each.

The Transformer is not magic. It is matrix multiplication, softmax, and a clever way of letting every word in a sentence talk to every other word in parallel. Once you see the recipe, you will recognize it everywhere.

## Prerequisites

- **Neural network basics:** You should know what a layer, activation function, and forward pass are. We are not starting from scratch.
- **Matrix multiplication:** Attention is all about multiplying matrices. If you remember how rows dot-product with columns, you are ready.
- **Softmax:** Converts a vector of numbers into probabilities that sum to 1. Used everywhere in attention.
- **Embeddings:** The idea that words can be represented as dense vectors (e.g., [0.2, -0.5, 0.9]). The Transformer learns these during training.
- **Tokenization (brief):** Splitting text into smaller units. We will cover this properly below.
- **RNNs and their limits (high-level):** Sequential processing, vanishing gradients, no parallelism. You do not need to implement one, just know why they were slow and limited.

---

## Concept 1: Why We Needed Transformers

### The problem it solves

Before 2017, sequence models were dominated by RNNs and LSTMs. They processed text one token at a time, left to right, updating a hidden state at each step. This had five crippling issues:

1. **Sequential bottleneck:** You could not process token 100 until you finished token 99. No parallelism, slow training.
2. **Fixed-length context bottleneck:** Encoder-decoder models compressed an entire input sequence into one fixed-size vector. Long sentences lost information.
3. **Vanishing gradients:** Gradients had to flow back through hundreds of sequential steps, shrinking or exploding along the way. Early tokens barely learned from later ones.
4. **Poor long-range dependencies:** Information from token 1 had to pass through token 2, 3, 4, ..., 50 to reach token 50. By the time it arrived, it was diluted.
5. **Uniform attention:** No mechanism to selectively focus on important words. Every token was weighted equally.

Imagine trying to translate "The animal did not cross the street because it was too tired" from English to French. By the time the decoder reaches "it," the encoder has long forgotten which noun ("animal" or "street") is relevant. The context vector is a lossy summary.

The Transformer solved all five problems at once by removing recurrence entirely and replacing it with attention.

### The intuition

Here is the core idea: instead of processing tokens one at a time in sequence, process all tokens simultaneously and let each token directly compare itself to every other token in one step. No sequential dependency. No bottleneck. Just parallel matrix operations.

Think of a classroom. In an RNN, each student passes a note to the next student, who adds their thoughts and passes it along. By the time the note reaches the back of the room, the first student's message is buried. In a Transformer, every student shouts their message at the same time, and every other student listens to everyone and decides who to pay attention to. It is noisy, but everyone hears everyone directly.

This direct connection means:
- **Parallelism:** All tokens processed at once. GPUs love this.
- **O(1) dependency path:** Any two tokens can influence each other in one attention step, regardless of distance.
- **Selective focus:** Attention weights decide which tokens matter. "it" can attend strongly to "animal" and weakly to "street."

### Common misunderstandings

- **"Transformers have no sequential processing at all."** False. Positional encodings inject order. Decoders use causal masking to enforce left-to-right generation. The architecture is parallelizable, but order still matters.
- **"Attention replaces all computation."** No. Attention is one component. Each Transformer layer also has a feed-forward network (two fully-connected layers with ReLU) applied position-wise.
- **"Transformers are always better than RNNs."** For most NLP tasks, yes. But Transformers have quadratic memory cost in sequence length (because every token attends to every other token). RNNs can handle arbitrarily long sequences with constant memory per step.

### Exam angle

MCQs will ask: "What is the primary advantage of the Transformer over RNNs?" The answer is **parallelism** and **direct O(1) connections between any two tokens**. Traps include "Transformers have fewer parameters" (false), "Transformers do not need training data" (nonsense), or "Transformers eliminate the need for embeddings" (no, they still use them).

---

## Concept 2: Tokenization

### The problem it solves

Models operate on numbers, not raw text. Tokenization converts a string like "unhappiness" into a sequence of integer IDs that the model can process. The challenge: what are the units? Words? Characters? Something in between?

- **Word-level:** "unhappiness" is one token. But then "unhappy" and "happiness" are separate tokens, even though they share meaning. Rare words get an UNK token. Vocabulary explodes to hundreds of thousands of words.
- **Character-level:** "unhappiness" becomes "u", "n", "h", "a", ... 11 tokens. Sequences become very long. The model has to learn "un-" is a prefix every time.
- **Subword-level (best of both):** Split into meaningful chunks. "unhappiness" might become ["un", "happiness"] or ["un", "happy", "ness"]. Common words stay whole, rare words decompose. Vocabulary stays reasonable (e.g., 50,000 tokens).

Subword tokenization is the standard in all modern LLMs.

### The intuition

Imagine you are learning a new language. At first, you memorize whole phrases. But when you encounter a new word like "reconfiguration," you do not panic. You recognize "re-" (again), "configure" (set up), "-ation" (noun suffix). Subword tokenization does the same thing: it breaks rare or compound words into familiar pieces.

Three popular algorithms:

1. **Byte Pair Encoding (BPE):** Start with characters. Merge the most frequent pair iteratively. "th" appears often in English, so merge it into one token. Then "th" + "e" becomes "the." Repeat 50,000 times. GPT-2 and GPT-3 use BPE.

2. **WordPiece:** Similar to BPE, but merges are chosen to maximize likelihood on training data, not raw frequency. BERT uses WordPiece.

3. **SentencePiece:** Treats the input as a raw byte stream, no pre-tokenization into words. Works for languages without spaces (Chinese, Japanese). Can learn BPE or unigram subword models. Used by T5, LLaMA, and many multilingual models.

### Worked example

Let's tokenize "unhappiness" with a tiny BPE vocabulary:

1. Start with characters: `['u', 'n', 'h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's']`
2. Training found "pp" is common, so we have a merge rule: "pp" → "pp". Apply it: `['u', 'n', 'h', 'a', 'pp', 'i', 'n', 'e', 's', 's']`
3. "un" is a common prefix, merge it: `['un', 'h', 'a', 'pp', 'i', 'n', 'e', 's', 's']`
4. "happ" might be a learned token: `['un', 'happ', 'i', 'n', 'e', 's', 's']`
5. "iness" is a common suffix: `['un', 'happ', 'iness']`

Final tokenization: 3 tokens instead of 11 characters or 1 giant word.

Now map each token to an integer ID from the vocabulary:
- "un" → 1247
- "happ" → 5890
- "iness" → 3021

The model sees `[1247, 5890, 3021]`.

### Common misunderstandings

- **"Tokenization is deterministic and universal."** No. Each model has its own vocabulary. GPT-2's tokenizer will split "unhappiness" differently than T5's. You must use the tokenizer that matches your model.
- **"A token is always a word."** No. A token can be a subword, a whole word, a punctuation mark, or even a space. In GPT-2, a leading space is part of the token (e.g., " unhappy" vs "unhappy" are different tokens).
- **"Tokenization is trivial preprocessing."** Wrong. Tokenization can introduce bias (e.g., non-English text is over-segmented), limit context window (short words take fewer tokens), and affect out-of-vocabulary behavior.

### Exam angle

"Why do modern LLMs use subword tokenization instead of word-level?" The answer is to **balance vocabulary size and sequence length, and handle rare or compound words gracefully**. A trap answer is "subword tokens are faster to compute" (not the reason).

Another MCQ: "What is the BPE merge criterion?" Answer: **merge the most frequent pair of tokens at each step** (for standard BPE; WordPiece uses likelihood).

---

## Concept 3: Embeddings and Positional Encoding

### The problem it solves

After tokenization, you have a sequence of integers, e.g., `[1247, 5890, 3021]`. The model needs dense vectors. Embeddings convert each token ID to a learned vector (e.g., 512-dimensional). But these vectors have no notion of position: the embedding for "cat" is the same whether it appears at position 0 or position 50. Attention is **permutation-invariant**: shuffle the tokens, and the attention weights do not change. We need to inject order.

Positional encodings are added to token embeddings to give each position a unique "fingerprint."

### The intuition (embeddings)

An embedding is a lookup table. Token ID 1247 maps to a 512-dimensional vector learned during training. Similar words end up with similar vectors (in some subspace) because they appear in similar contexts. This is standard word2vec / GloVe intuition, but the embeddings are learned jointly with the rest of the model, so they capture task-specific meaning.

### The intuition (positional encoding)

Imagine you are at a concert, and everyone is wearing the same uniform. You cannot tell who is in row 1 and who is in row 50. Now give each person a unique bracelet color based on their row number. Suddenly you can identify position.

Positional encodings are that bracelet. They are vectors added element-wise to the token embeddings. The Transformer uses a sinusoidal formula:

- Even dimensions: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`
- Odd dimensions: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`

where `pos` is the position index (0, 1, 2, ...), `i` is the dimension index (0, 1, 2, ..., d/2), and `d` is the embedding dimension (e.g., 512).

Why sinusoids? Four properties:

1. **Deterministic:** No parameters to learn. Works for any sequence length.
2. **Unique per position:** Each position gets a different vector.
3. **Relative distances:** The dot product between PE(pos) and PE(pos+k) depends only on k, not pos. The model can learn "tokens 3 positions apart" as a pattern.
4. **Generalizes beyond training length:** If you train on sequences up to 512 tokens, you can still generate positional encodings for position 600 at test time.

### Worked example

Let's compute the positional encoding for position 1, dimension 0 and 1, with d=512.

- Dimension 0 (even): `PE(1, 0) = sin(1 / 10000^(0/512)) = sin(1 / 1) = sin(1) ≈ 0.841`
- Dimension 1 (odd): `PE(1, 1) = cos(1 / 10000^(0/512)) = cos(1) ≈ 0.540`

For position 2:
- Dimension 0: `sin(2) ≈ 0.909`
- Dimension 1: `cos(2) ≈ -0.416`

Each position has a unique pattern across all 512 dimensions. When you add these to the token embeddings, the model can distinguish "cat at position 0" from "cat at position 10."

### The formula (now with meaning)

**Token embedding:** `e_t = W_embed[token_id]`, where `W_embed` is a (vocab_size × d_model) learned matrix.

**Positional encoding:** `PE(pos, i)` is a fixed sinusoid as above.

**Final input to Transformer:** `x_t = e_t + PE(pos)`, element-wise addition.

Why addition, not concatenation? Addition is simpler and mixes positional information with semantic information in the same vector. Some modern models (e.g., RoFormer) use rotary positional encodings instead, which rotate the embedding vector by an angle proportional to position. Same idea, different implementation.

### Common misunderstandings

- **"Positional encodings are learned."** Not in the original Transformer. They are fixed sinusoids. Some variants (e.g., BERT) learn positional embeddings, but that requires a maximum sequence length baked into the model.
- **"Without positional encodings, the Transformer cannot process sequences."** It can process them, but it treats "The cat sat" and "sat cat The" identically. For tasks like translation or generation, that is fatal.
- **"Positional encodings are only added once at the input."** Correct. They are not added again in higher layers. The information propagates through residual connections.

### Exam angle

"Why are positional encodings added to the input embeddings in a Transformer?" Answer: **Because attention is permutation-invariant and has no built-in sense of token order**. Trap: "To normalize the input" (false), "To reduce parameter count" (irrelevant).

Another MCQ: "What happens if you remove positional encodings?" Answer: **The model loses all word-order information and treats any permutation of tokens identically.**

---

## Concept 4: Self-Attention (The Core)

### The problem it solves

You have a sequence of vectors (token embeddings + positional encodings). How does each token build a richer representation by looking at the context of the entire sequence? Self-attention is the mechanism that lets each token attend to every other token and compute a weighted average of their values.

### The intuition

Imagine you are reading the sentence "The animal did not cross the street because it was too tired." When you encounter the word "it," you mentally search backward to figure out what "it" refers to. You compare "it" with "animal" (high match, both are nouns), "street" (weaker match, location), "cross" (verb, low match). You implicitly assign weights and conclude "it" = "animal."

Self-attention formalizes this. Every token compares itself to every other token and computes a weighted sum of their information. The comparison is done via dot products in a learned vector space.

Three learned projections:
1. **Query (Q):** What am I looking for?
2. **Key (K):** What do I advertise to others?
3. **Value (V):** What information do I carry?

Think of a library:
- You walk in with a **query**: "books on Transformers."
- Each shelf has a **key** (a label): "NLP," "Computer Vision," "Databases."
- You compare your query to each key. The NLP shelf scores high, CV scores medium, Databases scores low.
- You take a weighted combination of the **values** (the actual books) from each shelf, weighted by the scores.

### The formula (now with meaning)

**Step 1: Project input to Q, K, V**

Given input matrix X (sequence_length × d_model), compute:
- Q = X W_Q
- K = X W_K
- V = X W_V

where W_Q, W_K, W_V are learned weight matrices (d_model × d_k). Typically d_k = d_model / num_heads (we will cover multi-head attention shortly).

**Step 2: Compute attention scores**

Scores = Q K^T (matrix multiplication)

Each entry scores(i, j) = dot product of query_i and key_j. High dot product means token i finds token j relevant.

**Step 3: Scale**

Scores_scaled = Scores / sqrt(d_k)

Why divide by sqrt(d_k)? When d_k is large, dot products can become very large, pushing softmax into saturation (gradients near zero). Scaling keeps gradients healthy.

**Step 4: Softmax**

Attention_weights = softmax(Scores_scaled, dim=-1)

Each row of Attention_weights sums to 1. These are the attention probabilities.

**Step 5: Weighted sum of values**

Output = Attention_weights V

Each token's output is a weighted combination of all value vectors.

**Full formula:**

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

This is called **scaled dot-product attention.**

### Worked example

Tiny example: 2 tokens ("animal", "it"), embedding dimension d=2.

Input X:
```
X = [[1, 0],   # "animal"
     [0, 1]]   # "it"
```

For simplicity, let W_Q = W_K = Identity (just use X directly as Q and K). Let W_V project to V = [[1, 1], [1, 0]].

**Step 1: Q, K, V**
```
Q = [[1, 0],
     [0, 1]]
K = [[1, 0],
     [0, 1]]
V = [[1, 1],
     [1, 0]]
```

**Step 2: Scores = Q K^T**
```
Scores = [[1*1 + 0*0,  1*0 + 0*1],
          [0*1 + 1*0,  0*0 + 1*1]]
       = [[1, 0],
          [0, 1]]
```

**Step 3: Scale (d_k=2, sqrt(2) ≈ 1.41)**
```
Scores_scaled = [[1/1.41, 0/1.41],
                 [0/1.41, 1/1.41]]
              ≈ [[0.71, 0],
                 [0, 0.71]]
```

**Step 4: Softmax**

Row 1: softmax([0.71, 0]) = [exp(0.71), exp(0)] / (exp(0.71) + exp(0)) ≈ [2.03, 1.00] / 3.03 ≈ [0.67, 0.33]

Row 2: softmax([0, 0.71]) ≈ [0.33, 0.67]

(In the lecture slides, they use [0.73, 0.27] and [0.27, 0.73] because they skipped scaling or used a different normalization. The idea is the same.)

Attention_weights ≈ [[0.73, 0.27],
                     [0.27, 0.73]]

**Step 5: Output = Attention_weights × V**

Row 1 ("animal"): 0.73 × [1, 1] + 0.27 × [1, 0] = [1.0, 0.73]

Row 2 ("it"): 0.27 × [1, 1] + 0.73 × [1, 0] = [1.0, 0.27]

Notice: "animal" output has high second dimension (0.73) because it attended to itself ([1,1]). "it" output has lower second dimension (0.27) because it also attended to itself but less strongly.

### Common misunderstandings

- **"Query, Key, and Value are different inputs."** No. They all start from the same input X. The difference is in the learned projection matrices W_Q, W_K, W_V.
- **"Attention is just a weighted average."** True, but the weights are computed dynamically based on the input, not fixed. The model learns which tokens should attend to which.
- **"Self-attention is slow."** Computationally, it is O(n^2 d) where n is sequence length and d is dimension. For long sequences (thousands of tokens), this is expensive. Many modern models use sparse attention or sliding windows to reduce cost.

### Exam angle

"What is the role of the scaling factor sqrt(d_k) in scaled dot-product attention?" Answer: **To prevent dot products from becoming too large, which would cause softmax gradients to vanish**. Trap: "To normalize the output" (softmax already does that), "To speed up computation" (not the reason).

Another MCQ: "In self-attention, what do Q, K, and V represent?" Answer: **Query is what the token is looking for, Key is what each token advertises, Value is the information to mix**. Trap: "Q is input, K is hidden state, V is output" (nonsense).

---

## Concept 5: Multi-Head Attention

### The problem it solves

A single attention head captures one type of relationship. But language has many relationships: coreference ("it" → "animal"), syntax (verb → object), semantics (synonym relationships), positional (nearby tokens). One set of Q/K/V matrices cannot capture all of these simultaneously.

Multi-head attention runs multiple attention mechanisms in parallel, each with its own Q/K/V projections. Then concatenate the outputs and project back.

### The intuition

Imagine you are analyzing a sentence from multiple perspectives. One perspective focuses on coreference, another on verb-object pairs, another on sentiment words, another on positional neighbors. You run all these analyses in parallel, then combine the insights.

In the lecture slides, they show a 2-head example where Head 1 learns coreference ("it" → "animal") and Head 2 learns self-awareness ("it" → itself). Together, the model knows "it = animal, but it is also a distinct entity."

### The formula (now with meaning)

**Multi-head attention:**

MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) W_O

where each head is:

head_i = Attention(Q W_Q^(i), K W_K^(i), V W_V^(i))

and W_O is a learned output projection.

**Why does this work?**

Each head has its own projection matrices W_Q^(i), W_K^(i), W_V^(i). These learn different patterns. Head 1 might learn to attend to the previous token, Head 2 to nouns, Head 3 to punctuation, and so on. The model discovers these patterns automatically during training.

### Worked example (conceptual)

In the lecture, they show a 2-head example with d=2, each head dimension = 1.

**Head 1 (coreference):**
- Projection matrices make both "animal" and "it" query for the same key (first dimension).
- Result: Both tokens attend to "animal" with high weight. Coreference resolved.

**Head 2 (self-awareness):**
- Projection matrices focus on the second dimension.
- "animal" attends equally to itself and "it" (neutral).
- "it" attends more to itself (self-aware).

**Concatenation:**
- "animal" output: [1.46 from Head 1, 1.50 from Head 2] = [1.46, 1.50]
- "it" output: [1.46 from Head 1, 2.19 from Head 2] = [1.46, 2.19]

The second component (2.19 vs 1.50) shows "it" is more self-referential, while the first component (1.46, 1.46) shows both tokens share coreference to "animal."

### Common misunderstandings

- **"Each head looks at a different subset of tokens."** No. Every head attends to all tokens. The difference is in the learned Q/K/V projections, not the input.
- **"More heads are always better."** Not necessarily. 8 heads is typical. Beyond 16, diminishing returns. More heads mean smaller d_k per head, which can hurt expressiveness.
- **"Heads are interpretable."** Sometimes. Some heads learn clean patterns (e.g., attend to the previous token). Others are messy and hard to interpret. Do not expect every head to have a "meaning."

### Exam angle

"Why do Transformers use multi-head attention instead of single-head attention?" Answer: **To capture different types of relationships in parallel**. Trap: "To reduce computation" (false, multi-head uses the same or more FLOPs), "To allow different sequence lengths" (irrelevant).

---

## Concept 6: Masked (Causal) Self-Attention

### The problem it solves

In a decoder (e.g., GPT), the model generates text one token at a time, left to right. During training, we have the full target sequence, so we could process all tokens in parallel—but we must ensure that token i does not attend to tokens i+1, i+2, ... (future tokens). Otherwise, the model would cheat: when predicting token i, it would have already seen the answer.

Masked self-attention (also called causal attention) enforces this left-to-right constraint by setting future attention scores to -∞ before softmax, so they contribute zero weight.

### The intuition

Imagine you are taking a multiple-choice exam and the answer key is printed on the bottom of each page. If you can see the answer key while answering, you will cheat. To prevent this, the exam proctor covers up the answer key for question 3 while you are working on question 3. You can only see answers for questions 1 and 2 (which you have already answered).

Masked self-attention is that proctor. Token 3 can attend to tokens 1, 2, and 3, but not 4, 5, ... The mask is a lower-triangular matrix of True/False, where False means "block this position."

### Worked example

Sequence: "I love NLP <EOS>"

Attention scores (before mask):
```
         I    love  NLP  <EOS>
I        0.5  0.3   0.1  0.1
love     0.2  0.6   0.1  0.1
NLP      0.1  0.2   0.5  0.2
<EOS>    0.1  0.2   0.3  0.4
```

Apply causal mask (set upper triangle to -∞):
```
         I    love  NLP  <EOS>
I        0.5  -∞    -∞   -∞
love     0.2  0.6   -∞   -∞
NLP      0.1  0.2   0.5  -∞
<EOS>    0.1  0.2   0.3  0.4
```

After softmax, -∞ becomes 0:
```
         I    love  NLP  <EOS>
I        1.0  0.0   0.0  0.0
love     0.25 0.75  0.0  0.0
NLP      0.20 0.25  0.55 0.0
<EOS>    0.10 0.20  0.30 0.40
```

Token "I" can only attend to itself. Token "love" attends to "I" and "love." Token "<EOS>" attends to all past tokens. This ensures autoregressive generation.

### Common misunderstandings

- **"The mask is applied to the input."** No. The mask is applied to the attention scores (after Q K^T, before softmax).
- **"Masked attention is slower."** No. The mask is a simple element-wise operation. Parallelism is preserved: all tokens process simultaneously, but each token's attention is restricted to past positions.
- **"Masked attention is only used during training."** No. It is also used during generation. At each step, the decoder attends to all previous tokens (which have been generated so far) but not to future tokens (which do not exist yet).

### Exam angle

"Why is a causal mask used in Transformer decoders?" Answer: **To prevent tokens from attending to future positions during training, ensuring the model learns left-to-right generation**. Trap: "To reduce memory usage" (not the reason), "To make training faster" (it does not).

---

## Concept 7: Cross-Attention

### The problem it solves

In an encoder-decoder model (e.g., machine translation), the encoder processes the source sentence (English), and the decoder generates the target sentence (French). How does the decoder know which source words to focus on at each step? Cross-attention connects the two sequences: the decoder queries the encoder's hidden states.

### The intuition

You are translating "The cat sat on the mat" to French. When you generate the French word "chat" (cat), you need to attend to the English word "cat" in the source. Cross-attention lets the decoder ask, "Which source words are relevant to the word I am currently generating?"

The key difference from self-attention: Q (query) comes from the decoder, but K (key) and V (value) come from the encoder. Two different sequences interact.

### The formula (now with meaning)

Cross-Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc

- Q_dec: decoder hidden states (what am I generating?)
- K_enc, V_enc: encoder hidden states (what is available in the source?)

**Step-by-step:**

1. Decoder token i computes a query vector q_i.
2. Compare q_i to every encoder key k_j via dot product. High score means "encoder token j is relevant to decoder token i."
3. Softmax the scores to get attention weights.
4. Weighted sum of encoder values v_j.

### Worked example (conceptual)

English: "The cat sat on the mat."
French: "Le chat"

When generating "chat":
- Q: "chat" (decoder state)
- K, V: all encoder states ("The", "cat", "sat", "on", "the", "mat")

Attention scores:
- "The" → low
- "cat" → high (0.70)
- "sat" → medium (0.20)
- "on" → low
- "the" → low
- "mat" → low

Attention weights: [0.05, 0.70, 0.20, 0.03, 0.01, 0.01]

Output: 0.70 × V_cat + 0.20 × V_sat + small contributions from others.

The model learns to attend to "cat" when generating "chat."

### Common misunderstandings

- **"Cross-attention is used in BERT."** No. BERT is encoder-only and uses only self-attention. Cross-attention is used in encoder-decoder models like T5, BART, and the original Transformer.
- **"Cross-attention is bidirectional."** The encoder hidden states are bidirectional (the encoder saw the full source sentence), but the decoder queries are causal (the decoder only sees past target tokens).
- **"Cross-attention is optional."** For encoder-decoder models, no. It is the bridge between encoder and decoder. Without it, the decoder cannot access the source.

### Exam angle

"In cross-attention, where do Q, K, and V come from?" Answer: **Q from decoder, K and V from encoder**. Trap: "All three from encoder" (that is self-attention), "All three from decoder" (also self-attention).

---

## Concept 8: Residual Connections and Layer Normalization

### The problem it solves

Deep networks suffer from two issues: vanishing gradients (gradients shrink as they backpropagate through many layers) and training instability. Residual connections and layer normalization fix both.

### The intuition

**Residual connection:** Instead of x → layer → output, do x → layer → output + x. The input is added back to the output. This creates a "shortcut" for gradients to flow backward. Even if the layer learns nothing (output = 0), the gradient can still flow through the shortcut.

Think of a highway with an exit ramp. If the exit is closed, traffic can still flow on the main highway. Residuals are the main highway.

**Layer normalization:** Normalize the activations across the feature dimension (not the batch). Keeps activations in a reasonable range, stabilizes training. Applied after each sub-layer (attention, feed-forward).

### The formula (now with meaning)

**Residual connection:**
output = Layer(x) + x

**Layer normalization:**
LayerNorm(x) = (x - mean(x)) / sqrt(var(x)) × gamma + beta

where mean and var are computed across the feature dimension (d_model), and gamma and beta are learned scaling and shifting parameters.

**In a Transformer layer:**
```
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + FeedForward(x))
```

This is called "post-norm" (normalize after adding). Some models use "pre-norm" (normalize before the sub-layer), which is more stable for very deep models.

### Common misunderstandings

- **"Residual connections are only used in Transformers."** No. They come from ResNets (computer vision, 2015). Transformers adopted them.
- **"Layer norm is the same as batch norm."** No. Batch norm normalizes across the batch dimension (unstable for variable-length sequences). Layer norm normalizes across features.
- **"Residuals allow infinite depth."** Not quite. Very deep Transformers (100+ layers) still face optimization challenges. But residuals and layer norm allow depths of 12–24 layers to train reliably.

### Exam angle

"Why are residual connections used in Transformers?" Answer: **To allow gradients to flow through shortcut paths, preventing vanishing gradients in deep networks**. Trap: "To reduce parameter count" (residuals add no parameters), "To increase accuracy" (they enable training, not accuracy directly).

---

## Concept 9: Feed-Forward Network

### The problem it solves

Attention mixes information across tokens. But it does not add much non-linear transformation within each token's representation. The feed-forward network (FFN) is a simple two-layer MLP applied independently to each token. It adds expressiveness.

### The intuition

Think of attention as "communication" (tokens talking to each other) and FFN as "computation" (each token thinking about what it heard). After attention, each token has gathered context from the entire sequence. Now it needs to process that context and transform it.

### The formula (now with meaning)

FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

where W_1 projects from d_model to d_ff (typically d_ff = 4 × d_model, e.g., 512 → 2048), ReLU adds non-linearity, and W_2 projects back to d_model.

This is applied position-wise: the same FFN is applied to every token independently. No interaction between tokens in this step.

### Common misunderstandings

- **"FFN is a Transformer-specific invention."** No. It is a standard MLP. The novelty is that it is applied per-token, not per-sequence.
- **"FFN is where the parameters are."** Partially true. FFN accounts for about 2/3 of the parameters in a Transformer layer (because d_ff is large). Attention accounts for the other 1/3.

### Exam angle

"What is the purpose of the feed-forward network in a Transformer layer?" Answer: **To apply non-linear transformations to each token's representation independently**. Trap: "To perform attention" (that is the attention layer), "To reduce sequence length" (FFN does not change length).

---

## Concept 10: The Full Transformer Encoder

### Putting it together

An encoder block consists of:
1. Multi-head self-attention (all tokens attend to all tokens)
2. Residual connection + layer norm
3. Feed-forward network (per-token MLP)
4. Residual connection + layer norm

Stack N of these blocks (N=6 in the original Transformer, N=12 in BERT-Base, N=24 in BERT-Large).

**Input:** Token embeddings + positional encodings
**Output:** Contextualized representations (each token now carries information from the entire sequence)

### Encoder-only models (BERT)

BERT uses only the encoder. It processes the full input in parallel (bidirectional). The output is used for tasks like classification, NER, question answering.

**Key point:** Encoder-only models cannot generate text autoregressively (they see the whole sequence at once, no causal masking). They are used for understanding tasks.

---

## Concept 11: The Full Transformer Decoder

### Putting it together

A decoder block consists of:
1. Masked multi-head self-attention (causal, attends only to past tokens)
2. Residual connection + layer norm
3. Cross-attention (query from decoder, key/value from encoder)
4. Residual connection + layer norm
5. Feed-forward network
6. Residual connection + layer norm

Stack N of these blocks.

**Input:** Target token embeddings + positional encodings (during training: full target sequence with causal mask; during generation: one token at a time)
**Output:** Contextualized representations → linear layer → softmax over vocabulary → next token probabilities

### Decoder-only models (GPT)

GPT uses only the decoder. No encoder, no cross-attention. The decoder is trained to predict the next token given all previous tokens (causal language modeling).

**Key point:** Decoder-only models are generative. They autoregressively produce text one token at a time.

### Encoder-decoder models (T5)

T5 uses both encoder and decoder. The encoder processes the input (e.g., English sentence), the decoder generates the output (e.g., French sentence). Cross-attention connects them.

**Key point:** Encoder-decoder models are best for sequence-to-sequence tasks (translation, summarization).

---

## Concept 12: Autoregressive Generation

### The problem it solves

At inference time, how does the decoder generate text? It starts with a special token (e.g., <BOS> or a prompt), predicts the next token, appends it, and repeats.

### The intuition

You are writing a sentence word by word, and at each step you decide the next word based on everything you have written so far. That is autoregressive generation.

### The process

1. Start with `<BOS>` or prompt.
2. Run the decoder (with causal mask).
3. The output logits are projected to vocabulary size (e.g., 50,000).
4. Apply softmax → token probabilities.
5. Sample or argmax to pick the next token.
6. Append the token to the sequence.
7. Repeat until `<EOS>` or max length.

**Greedy decoding:** Always pick the highest-probability token (argmax). Deterministic. Can get stuck in repetitive loops.

**Beam search:** Keep top-k sequences at each step, expand, and prune. More expensive, better quality.

**Sampling:** Pick a token randomly according to the probability distribution. Adds randomness. Temperature and top-p are sampling strategies (we will cover these in Lecture 4).

### Common misunderstandings

- **"Autoregressive generation is slow."** Yes. You must run the decoder N times to generate N tokens. Cannot parallelize across tokens (each token depends on the previous). This is why GPUs batch multiple sequences together.
- **"Autoregressive models cannot be trained in parallel."** False. During training, you have the full target sequence. Use teacher forcing: feed the ground-truth tokens (with causal mask) and predict the next token for all positions simultaneously. Parallel training, sequential generation.

### Exam angle

"What is the difference between training and generation in an autoregressive model?" Answer: **Training uses the full target sequence with causal masking (parallel), generation produces tokens one at a time (sequential)**. Trap: "Training is sequential" (false).

---

## Concept 13: Encoder-Only vs Decoder-Only vs Encoder-Decoder

### When to pick each

**Encoder-only (BERT, RoBERTa):**
- **Use case:** Classification, NER, QA, any task where you need to understand the input but not generate long text.
- **Why:** Bidirectional context (no causal mask), sees the full input at once.
- **Cannot:** Generate text autoregressively.

**Decoder-only (GPT, LLaMA, Claude):**
- **Use case:** Text generation, chat, code generation, any autoregressive task.
- **Why:** Causal masking, trained to predict the next token.
- **Can:** Generate arbitrarily long text. Can also do classification by treating it as generation (e.g., "Sentiment: positive").

**Encoder-decoder (T5, BART, original Transformer):**
- **Use case:** Translation, summarization, any task where input and output are clearly separated.
- **Why:** Encoder builds bidirectional representation of input, decoder generates output conditioned on encoder.
- **Trade-off:** More parameters (two separate stacks), but cleaner separation.

### Common misunderstandings

- **"BERT is better than GPT."** No. They are designed for different tasks. BERT is better at understanding, GPT is better at generation.
- **"GPT cannot do classification."** False. You can prompt GPT with "Classify the sentiment: [text]. Answer: " and it will generate "positive" or "negative." But BERT is more efficient for this task because it does not need to generate.

### Exam angle

"Which Transformer architecture is best for machine translation?" Answer: **Encoder-decoder (T5, BART)**, because translation is a seq2seq task. Decoder-only can do translation but is less efficient. Encoder-only cannot generate.

Another MCQ: "What is the key difference between BERT and GPT?" Answer: **BERT is encoder-only and bidirectional, GPT is decoder-only and causal**. Trap: "BERT has more parameters" (not necessarily, depends on the variant).

---

## Putting it together

The Transformer replaced recurrence with attention, enabling parallelism and O(1) connections between any two tokens. Every modern LLM is built on this foundation: tokenize the input, add positional encodings, stack self-attention and feed-forward layers, add residual connections and layer norm. Encoder-only models (BERT) understand, decoder-only models (GPT) generate, encoder-decoder models (T5) translate. The next lecture will cover how these models are trained, how they decode, and how you use them via APIs.

---

## Check yourself (5 questions)

**Q1.** In a Transformer, what is the primary purpose of positional encodings?

A) To reduce the number of parameters in the model  
B) To allow the model to process tokens sequentially like an RNN  
C) To provide the model with information about token order, since attention is permutation-invariant  
D) To normalize the input before attention  

**Answer:** C — Attention has no built-in notion of order. Without positional encodings, "cat sat mat" and "mat sat cat" would be identical. The sinusoidal formula gives each position a unique vector.

---

**Q2.** Why is the scaling factor sqrt(d_k) used in scaled dot-product attention?

A) To speed up matrix multiplication  
B) To prevent large dot products from pushing softmax into saturation (vanishing gradients)  
C) To normalize the output to have mean 0 and variance 1  
D) To reduce memory usage  

**Answer:** B — When d_k is large, Q K^T values become large, making softmax outputs near 0 or 1 (gradients near zero). Dividing by sqrt(d_k) keeps values in a reasonable range.

---

**Q3.** In a Transformer decoder, a causal mask is applied to:

A) The input embeddings  
B) The attention scores (after Q K^T, before softmax)  
C) The output logits  
D) The positional encodings  

**Answer:** B — The mask sets future positions to -∞ in the score matrix, so softmax assigns them zero weight. This ensures token i cannot attend to tokens i+1, i+2, ... during training or generation.

---

**Q4.** Multi-head attention uses multiple attention heads to:

A) Process different subsets of tokens in parallel  
B) Reduce computational cost compared to single-head attention  
C) Capture different types of relationships (e.g., syntax, semantics, coreference) in parallel  
D) Allow variable sequence lengths  

**Answer:** C — Each head learns different Q/K/V projections, discovering different patterns. One head might learn coreference, another verb-object relations, etc. All heads attend to all tokens, but learn different patterns.

---

**Q5.** Which Transformer variant is best suited for autoregressive text generation?

A) BERT (encoder-only)  
B) GPT (decoder-only)  
C) Vision Transformer (encoder-only)  
D) T5 encoder (encoder part only)  

**Answer:** B — GPT is decoder-only with causal masking, designed for autoregressive generation. BERT is encoder-only (bidirectional, cannot generate). T5 encoder alone cannot generate either. T5 uses both encoder and decoder for generation.

---

## Quick reference (for last-day revision)

**Tokenization:** Subword (BPE, WordPiece, SentencePiece) splits text into learned units. Balances vocab size and sequence length.

**Embeddings:** Token ID → learned d-dimensional vector. Lookup table.

**Positional encoding:** PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...). Added to embeddings. Deterministic, unique per position.

**Self-attention:** Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. Q = what I want, K = what I offer, V = what I carry. O(n^2 d) complexity.

**Multi-head attention:** h parallel heads, each with own Q/K/V projections. Concat outputs, project with W_O. Captures diverse relationships.

**Masked attention:** Set future positions to -∞ before softmax. Ensures causal (left-to-right) generation in decoders.

**Cross-attention:** Q from decoder, K/V from encoder. Bridges encoder-decoder in seq2seq models.

**Residual connection:** output = Layer(x) + x. Shortcut for gradient flow.

**Layer norm:** Normalize across features. Stabilizes training.

**Feed-forward:** Two-layer MLP per token. FFN(x) = ReLU(x W_1) W_2. d_ff = 4 × d_model typically.

**Encoder block:** Self-attention → add & norm → FFN → add & norm.

**Decoder block:** Masked self-attention → add & norm → cross-attention → add & norm → FFN → add & norm.

**Encoder-only (BERT):** Bidirectional, for understanding tasks (classification, NER).

**Decoder-only (GPT):** Causal, for generation (text, code).

**Encoder-decoder (T5):** Both. For seq2seq (translation, summarization).

**Autoregressive generation:** Start with <BOS>, predict next token, append, repeat. Training is parallel (teacher forcing + causal mask), generation is sequential.

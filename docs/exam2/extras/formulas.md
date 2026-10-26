# Numbers & formulas

**Why this page exists.** Students who sat the first slot of Exam 2 reported **formula-based questions from the fine-tuning chapters**. The course slides do contain formulas with fully worked examples, so this page collects every one of them in one place. Each formula comes with what its letters mean, the slide's own worked example solved step by step, and the trap to avoid.

**How to use it.** Read a section, then cover the worked example and redo it yourself on paper. The arithmetic is never hard. The questions test whether you know **which numbers go where**. The practice questions at the bottom are all calculations.

**About sources.** Almost everything here is from the course slides, pre-reads and Q&A sheets. A few items come from the original research papers and are marked *(beyond the course slides)*.

!!! tip "Three habits that prevent most mistakes"
    **Write the units.** Bytes, bits, GB, parameters. Most wrong options are a units slip (bits vs bytes, or MB vs GB).

    **Know your rough sizes.** 1 GB ≈ 1 billion bytes, so a model with N billion parameters at 2 bytes each takes 2N GB.

    **Check the order of magnitude last.** If LoRA comes out larger than the full matrix, something went wrong.

---

## Part A · Fine-tuning

### 1. How much memory full fine-tuning needs

\[
M_{\text{full FT}} \approx N \times (2_{\text{weight}} + 2_{\text{gradient}} + 8_{\text{Adam}}) \text{ bytes} = 12N \text{ bytes}
\]

**What the letters mean.** N is the number of parameters being trained. The weight and its gradient are FP16 (2 bytes each). Adam keeps two running averages per weight in FP32 (4 bytes each, so 8 bytes).

**The slides' worked examples.**

- A **10B** model: 10 billion × 12 bytes = **120 GB**. The slide shows this spread across a pack of seven 24 GB GPUs, because activations need room too.
- A **70B** model: 70 billion × 12 bytes = **840 GB**, which is about 20 data-centre GPUs.

**Why you'll also see 16 bytes.** Mixed-precision training usually keeps an extra **FP32 master copy** of every weight (4 more bytes): 2 + 2 + 8 + 4 = **16 bytes per parameter**. That's where the slide line "Adam needs roughly 16 bytes per parameter" and the revision slide's **160 GB for 10B** come from. Both numbers are right; they just count differently. If a question states which parts to count, follow it.

**Just storing a model (no training).** Use bytes per weight only.

- FP16: N × 2 bytes. A 7B model is **14 GB**; a 13B model is **26 GB**.
- Many copies multiply: 50 fine-tuned 7B copies = 50 × 14 GB = **700 GB**. 200 client copies of a 13B model = 200 × 26 GB = **5.2 TB**.
- 4-bit (NF4): N × 0.5 bytes. A 7B model drops from 14 GB to about **3.5 GB**; a 13B model to about **7 GB**.

**Trap.** "The weights are the biggest part of training memory." No. Adam's states (8 bytes) are four times the FP16 weight (2 bytes).

### 2. Memory with LoRA and QLoRA

\[
M \approx N \times \frac{\text{bits per parameter}}{8} \text{ bytes}
\]

The slides express the cost per parameter of the **whole** model, then multiply by N.

**LoRA on 70B.** The frozen weight is still 16 bits. The gradients, optimizer states and adapter weights shrink to a fraction of a bit per parameter (about 0.4 + 0.8 + 0.4), because they exist only for the tiny A and B. That's about 17.6 bits = 2.2 bytes per parameter, and 70B × 2.2 bytes ≈ **154 GB** (about 4 GPUs).

**QLoRA on 70B.** The frozen weight drops to 4 bits. The slide quotes about **5.2 bits per parameter**, and 70B × 5.2 ÷ 8 ≈ 45.5, so about **46 GB**, which fits on **one** GPU. (The slide's listed parts add up to 5.6 bits, slightly more than its own 5.2 total. Use the slide's headline figure, 46 GB.)

**The revision slide's version.** A 10B model: full fine-tuning **160 GB** vs LoRA **40 GB**. It's the same pattern with a different counting assumption.

**Trap.** "LoRA saves memory because A and B are small files." The main saving is that the frozen weights need **no gradients and no Adam states**.

### 3. Data splits and the accuracy paradox

**Splitting.** The slide's default is 70 / 15 / 15. With 10,000 examples that's **7,000** train, **1,500** validation and **1,500** test.

**The accuracy paradox.** If 0.2% of transactions are fraud, a model that always answers "not fraud" is right 99.8% of the time. Its accuracy is **99.8%**, yet its recall on fraud is **0**, because it catches none.

### 4. Steps, epochs and warm-up

\[
\text{steps per epoch} = \frac{N_{\text{train}}}{\text{effective batch}} \qquad
\text{total steps} = \text{steps per epoch} \times \text{epochs} \qquad
\text{warm-up steps} = \text{warmup\_ratio} \times \text{total steps}
\]

**Worked examples (from the course Q&A).**

- 10,000 training examples with a batch size of 100 gives **100 steps per epoch**. Five epochs give **500 steps**.
- 1,000 total optimizer steps with `warmup_ratio = 0.03` gives **30 warm-up steps**.

**Trap.** Warm-up is counted in **optimizer steps**, not epochs. With gradient accumulation, one optimizer step happens only after several micro-batches, so always divide by the **effective** batch.

### 5. Effective batch size

\[
B_{\text{eff}} = B_{\text{micro}} \times G_{\text{accum}} \times N_{\text{GPU}}
\]

**What the letters mean.** B_micro is how many examples one GPU processes in one forward/backward pass. G_accum is how many micro-batches are added up before one update. N_GPU is the number of GPUs.

**The slide's worked example.** Micro-batch 2, gradient accumulation 16, 4 GPUs: 2 × 16 × 4 = **128**.

**Trap.** Gradient accumulation raises the effective batch **without** raising memory. Memory depends on the micro-batch.

### 6. The linear scaling rule for the learning rate

\[
\eta_{\text{new}} \approx \eta_{\text{old}} \times \frac{B_{\text{new}}}{B_{\text{old}}}
\]

**In words.** If the effective batch doubles, roughly double the learning rate.

**The slide's worked example.** Batch 32 at LR 2e-5 becomes batch 64 at LR **4e-5**. Going on to batch 128 (4× the original) would give 8e-5.

**Typical values to recognise.** Full fine-tuning peaks around 1e-5 to 2e-5. LoRA peaks around 1e-4 to 2e-4, about ten times higher.

### 7. Gradient clipping

\[
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
\]

**What the letters mean.** g is the gradient, ‖g‖ is its size (norm), and c is the cap, `max_grad_norm` (default **1.0**).

**Worked examples.**

- ‖g‖ = 4 and c = 1: the multiplier is min(1, 1/4) = 0.25. The gradient is **shrunk to a quarter**, so its norm becomes exactly 1, and its **direction is unchanged**.
- ‖g‖ = 0.5 and c = 1: the multiplier is min(1, 2) = 1. **Nothing changes**, because the gradient was already under the cap.

**Trap.** Clipping doesn't skip the step, and it doesn't cap each number separately. It rescales the whole gradient vector. If more than about 20% of steps get clipped, the learning rate is too high.

### 8. Weight averaging (WiSE-FT) against forgetting

\[
\theta_{\text{merged}} = (1-\alpha)\,\theta_{\text{base}} + \alpha\,\theta_{\text{ft}}
\]

**In words.** Blend each fine-tuned weight with the original one. α = 0 gives back the base model, α = 1 keeps the fine-tuned model, and anything in between trades task skill against general ability.

**Worked example.** A weight was 2.0 in the base model and 3.0 after fine-tuning. With α = 0.5 the merged weight is 0.5 × 2.0 + 0.5 × 3.0 = **2.5**. With α = 0.2 it's 0.8 × 2.0 + 0.2 × 3.0 = **2.2**, much closer to the base.

### 9. Adapter parameter count

\[
h' = h + W_{\text{up}}\, f(W_{\text{down}}\, h) \qquad \#\text{params per adapter} = 2md + m + d
\]

**What the letters mean.** d is the hidden size of the model. m is the adapter's bottleneck size (much smaller than d).

**Where 2md + m + d comes from.** The down-projection has d × m weights and m biases. The up-projection has m × d weights and d biases. Add them: md + m + md + d = **2md + m + d**.

**The slide's worked example.** With d = 768 and m = 64:

1. 2 × 64 × 768 = 98,304
2. Add the biases: 98,304 + 64 + 768 = **99,136, about 0.1 million**.
3. The slide counts one transformer decoder layer as about 16d² + 19d = 16 × 589,824 + 19 × 768 ≈ **9.5 million**.
4. So the adapter adds about 0.1M ÷ 9.5M ≈ **1%** per layer. (The Q&A quotes the layer as 9.75M. Either way, it's about 1%.)

**Trap.** The bottleneck doesn't permanently shrink the hidden size. The adapter squeezes d down to m and immediately expands back to d.

### 10. Soft-prompt parameter counts

**Prompt tuning.** It trains one matrix of virtual-token embeddings, W_p of size d × t, where t is the number of virtual tokens. That's **t × d** numbers. With t = 20 and d = 768: 20 × 768 = **15,360** parameters.

**Prefix tuning.** It learns key and value prefixes (P_k and P_v) at **every** layer, so it's about **2 × L × t × d**, where L is the number of layers. With L = 12, t = 10 and d = 768: 2 × 12 × 10 × 768 = **184,320**. This is why prefix tuning trains more than prompt tuning for the same number of virtual tokens. *(This count follows from the slide's design; the slide doesn't work it out.)*

**IDPG's prompt generator.** The slide gives its size as **m(d + 1) + td(m + 1)**. It's a down-projection from d to m (with biases) followed by an up-projection from m to the full prompt of t × d numbers (with biases). Recognise the shape of this formula; you're unlikely to need to evaluate it.

### 11. LoRA parameter count

\[
h = W_0 x + \frac{\alpha}{r} B A x \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k} \qquad \#\text{params} = r(d + k)
\]

**In words.** Full fine-tuning of one d × k matrix trains **d × k** numbers. LoRA trains B (d × r) and A (r × k), which is **r × (d + k)** numbers.

**Worked example 1: the class quiz.** One 1000 × 1000 matrix has 1,000,000 weights. A 1000 × 100 matrix followed by a 100 × 1000 matrix has 100,000 + 100,000 = **200,000**, five times fewer. That second network is a rank-100 factorisation.

**Worked example 2: one big matrix.** A 4096 × 4096 matrix has 16,777,216 weights. LoRA with r = 8 trains 8 × (4096 + 4096) = **65,536**, which is about **0.39%**.

**Worked example 3: the slide's LLaMA-13B adapter.** Here d = k = 5120, there are L = 40 layers, LoRA goes on M = 4 matrices per layer (Q, K, V and O), and r = 16. The formula is L × M × r × (d + k).

1. Per matrix: 16 × (5120 + 5120) = **163,840**
2. Per layer: 163,840 × 4 = **655,360**
3. Whole model: 655,360 × 40 = **26.2 million**
4. Size in FP16: 26.2 million × 2 bytes = **52.4 MB**, which is about **0.2%** of 13 billion

That's the "about 50 MB per client" in the 200-client story (Chapter 04).

**The α/r scaling.** The update is multiplied by α/r. With α = 16 and r = 8 the multiplier is 2. If you raise r to 16 and keep α = 16, it becomes 1. Dividing by r keeps the update's size stable when you change the rank.

**Traps.** Doubling r **doubles** LoRA's parameters (the count is linear in r). LoRA adds parameters during training (W₀ + A + B); it reduces only the number you **train**.

### 12. Absmax quantization (8-bit)

\[
c = \frac{127}{\max |X|} \qquad q = \text{round}(c \cdot X) \qquad \hat{X} = \frac{q}{c} \qquad \text{error} = |\hat{x} - x|
\]

**What the letters mean.** c is the quantization constant (the scale). 127 is the largest value an 8-bit signed integer is used for. q holds the stored integers, and X̂ is the dequantized approximation.

**The slide's worked example.** X = [0.5, −2.0, 1.2, 3.0].

1. The largest absolute value is 3.0, so c = 127 ÷ 3.0 ≈ **42.33**.
2. Multiply and round: 0.5 × 42.33 = 21.2 → **21**; −2.0 × 42.33 = −84.7 → **−85**; 1.2 × 42.33 = 50.8 → **51**; 3.0 × 42.33 = **127**.
3. Dequantize by dividing by c: about [0.496, −2.008, 1.205, 3.0].
4. The errors are tiny: about 0.004, 0.008, 0.005 and 0. The largest value is always reproduced exactly.

**Trap.** You store the integers **and** the constant c. Without c you can't convert back.

### 13. NF4: 4-bit storage and dequantization

**Levels.** 4 bits give 2⁴ = **16 levels**. NF4 places them at quantiles of a normal distribution, so more levels sit near zero. Two 4-bit codes fit in one byte.

**The slide's storage example.** Six FP32 weights take 6 × 4 = **24 bytes**. As 4-bit codes they take 6 × 0.5 = **3 bytes**, plus one 4-byte absmax constant, so **7 bytes** in total. The saving is (24 − 7) ÷ 24 ≈ **70.8%**.

**Dequantizing.** value ≈ absmax × NF4 level. In the slide's example the absmax is 1.65 and a weight of −1.2 was stored as the level −0.696, so it comes back as 1.65 × −0.696 ≈ **−1.149** (an error of about 0.05). The revision slide does the same with an absmax of 0.82: it divides every weight by 0.82 to scale them into [−1, 1], stores the nearest level, and multiplies by 0.82 to get back.

### 14. Block-wise quantization overhead and double quantization

\[
\text{overhead per parameter} = \frac{\text{bits per constant}}{\text{block size}}
\]

**Worked examples (slides).** With a 32-bit constant per block of **64** weights: 32 ÷ 64 = **0.5 bits per parameter**. With blocks of **128**: 32 ÷ 128 = **0.25 bits**. Smaller blocks are more precise but cost more overhead. Double quantization reduces this cost by quantizing the constants themselves.

**The slide's 25-value example.** It compares four ways to store 25 numbers.

1. **No quantization:** 25 × 4 bytes = **100 bytes**.
2. **8-bit, one constant for everything:** 25 × 1 byte + one 4-byte constant = **29 bytes**.
3. **8-bit, block-based (5 blocks of 5):** 25 bytes + five 4-byte constants (20 bytes) = **45 bytes**.
4. **Block-based plus double quantization:** 25 bytes + the five constants stored as 1 byte each (5 bytes) + one 4-byte constant for those = **34 bytes**.

The lesson is that block-wise storage is more accurate but adds constants, and double quantization wins most of that overhead back (45 → 34 bytes).

---

## Part A, continued · Alignment

### 15. The reward model's loss (Bradley–Terry)

\[
L_{RM} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
\]

**What the letters mean.** r(x, y_w) is the reward model's score for the human-preferred (chosen) answer, and r(x, y_l) is its score for the rejected one. σ is the sigmoid, which turns the gap into a number between 0 and 1.

**Worked examples.**

- **Chosen scores 2 higher (gap +2):** σ(2) ≈ 0.88, so the loss ≈ 0.13. **Small**, which is what we want.
- **Equal scores (gap 0):** σ(0) = 0.5, so the loss = −log 0.5 ≈ 0.69.
- **Rejected scores 2 higher (gap −2):** σ(−2) ≈ 0.12, so the loss ≈ 2.13. **Large**, so training pushes hard to fix it.

**Trap.** The minus sign is there because training **minimises** loss; it turns "make the chosen probability high" into "make this number small".

### 16. The KL-penalised reward (RLHF)

\[
r_{\text{total}} = r_\phi(x, y) - \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
\]

**In words.** The reward the policy actually receives is the reward model's score minus a penalty for drifting away from the frozen reference model.

**Worked example.** The reward model gives 8, the KL divergence from the reference is 4, and β = 0.5. The total reward is 8 − 0.5 × 4 = **6**. Double β to 1.0 and the total drops to 4. A bigger β pulls the model **closer to the reference**.

### 17. PPO: probability ratio, clipping and advantage

\[
\text{ratio} = \frac{\pi_{\text{new}}(a \mid s)}{\pi_{\text{old}}(a \mid s)} \qquad A = Q - V
\]

**The ratio.** It shows how much more (or less) likely the updated model makes an action than the model that generated the sample. If a token's probability goes from 0.30 to 0.45, the ratio is 0.45 ÷ 0.30 = **1.5**.

**Clipping.** PPO keeps the ratio inside **[1 − ε, 1 + ε]**, usually with ε = 0.2, so inside [0.8, 1.2]. The ratio of 1.5 above counts only as **1.2**. That's the "don't change too much in one update" rule. *(The exact clipping range is from the PPO paper, beyond the course slides.)*

**Advantage.** A = Q − V means "how much better this action was than the average from this state". With Q = 7 and V = 5, A = **+2**, so the action gets reinforced. A negative advantage makes the action less likely.

**Counting models.** PPO needs **4** models: the policy and the value model are trained, and the reference and reward models are frozen. DPO needs **2**: the policy (trained) and the reference (frozen).

### 18. The DPO loss

\[
L_{DPO} = -\log \sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)
\]

**In words.** Compare how much the model has raised the chosen answer's probability (relative to the reference) with how much it has raised the rejected answer's. The loss falls as the chosen answer gains **relative to** the rejected one. β sets how hard this pushes (the hands-on used β = 0.1).

**Worked example.** The reference gives the chosen and rejected answers 0.40 each. After training, the model gives the chosen answer 0.60 and the rejected answer 0.20.

1. Chosen: log(0.60 ÷ 0.40) = log 1.5 ≈ +0.41
2. Rejected: log(0.20 ÷ 0.40) = log 0.5 ≈ −0.69
3. Difference: 0.41 − (−0.69) = 1.10. Times β = 0.1 gives 0.11.
4. σ(0.11) ≈ 0.53, so the loss ≈ **0.64**. Before training (model = reference) the difference is 0 and the loss is **0.69**. It went down, as it should.

**Trap.** The revision slides' example numbers (0.60 → 0.30 for the chosen answer) contradict the direction DPO pushes. Remember the direction, not those numbers: **chosen up, rejected down, relative to the reference.**

---

## Part B · Vision-language models

### 19. How many patches a ViT makes

\[
N_{\text{patches}} = \frac{H \times W}{P^2}
\]

**What the letters mean.** H and W are the image height and width in pixels. P is the patch size.

**Worked examples.**

- 224 × 224 with 16 × 16 patches: (224 ÷ 16)² = 14² = **196** patches (197 with a [CLS] token).
- 224 × 224 with 14 × 14 patches (CLIP ViT-L/14): (224 ÷ 14)² = 16² = **256** patches.
- 336 × 336 with 14 × 14 patches (LLaVA-1.5's 336 px encoder): (336 ÷ 14)² = 24² = **576** patches.

**Trap.** Halving the patch size **quadruples** the patch count. Qwen-VL always outputs **256 visual tokens**, whatever the patch count, because its 256 learned queries summarise the patches.

### 20. CLIP's similarity scores and temperature

\[
\text{logits} = \frac{I_e\, T_e^{\top}}{\tau}
\]

**What the letters mean.** I_e holds the batch's image embeddings and T_e the text embeddings, all normalised to length 1, so each entry is a **cosine similarity**. τ is the learned temperature.

**Counting pairs.** A batch of N pairs gives an N × N grid of scores. Each image has **1** positive and **N − 1** negatives. CLIP's batch of 32,768 gives each image **32,767** negatives and about 1.07 billion scores per batch.

**Worked example of temperature.** Two captions score cosine 0.30 and 0.25 against an image.

- With τ = 1 the logits are 0.30 and 0.25, and the softmax is nearly a coin flip (about 51% vs 49%).
- With τ = 0.01 the logits become 30 and 25, and the softmax gives about **99%** to the first caption.

A lower τ makes the choice sharper.

### 21. SAM's numbers

**Loss.** 20 × focal loss + 1 × Dice loss for the masks (a **20 : 1** weighting), plus a mean-squared-error loss for the IoU head. *(The 20 : 1 weighting is from the SAM paper.)*

**Speed.** The image encoder runs **once** (about 0.15 s). Each prompt then costs only the light decoder (about 50 ms). So ten clicks on one image cost about 0.15 + 10 × 0.05 = **0.65 s**, not 10 × 0.2 s.

**Data engine.** The fully automatic stage prompts each image with a **32 × 32 grid** = **1,024 points**. SA-1B has about **1.1 billion masks on 11 million images**, about 100 masks per image. Each prompt returns **3** candidate masks.

### 22. Retrieval scores: MaxSim, Recall@K and indexing cost

\[
s(q, d) = \sum_{i} \max_{j}\; q_i \cdot d_j
\]

**MaxSim in words.** For each query token i, find its best-matching page patch j, then add up those best scores.

**Worked example.** The query has 2 tokens and the page has 3 patches.

- Token 1 scores [0.2, **0.9**, 0.1] against the three patches, so its best is 0.9.
- Token 2 scores [**0.7**, 0.3, 0.4], so its best is 0.7.
- The page's score is 0.9 + 0.7 = **1.6**.

**Recall@K.** If the correct item is in the top 5 for 80 of 100 test queries, Recall@5 = 80 ÷ 100 = **0.80**. Recall@5 is never lower than Recall@1.

**ColPali's indexing speed.** 0.39 s per page vs 7.22 s per page for the OCR pipeline, which is about 7.22 ÷ 0.39 ≈ **18×** faster. For 1,000 pages that's about 6.5 minutes vs about 2 hours.

---

## Parts C and D · Agents and Responsible AI

The agent and Responsible AI sessions have **no formulas**. Their numbers are design rules to recognise, such as "timeouts: tool < node < run" in LangGraph, or capping an agent loop with a maximum number of iterations.

---

## Practice questions

All of these are calculations. Work each one out on paper before you pick an option.

<div class="practice-quiz" data-quiz="formulas">
  <p><em>Loading practice questions…</em></p>
</div>

# Lecture 2 — Transformer Architecture, Attention, Tokenization
**Instructor:** Prof. Niloy Ganguly, IIT Kharagpur | **Module:** Advanced Architectures

## TL;DR (5 bullets max)
1. CNNs use LOCAL CONNECTIVITY (neurons connect to small regions, not entire input) + parameter sharing (same filter reused) → this is the structural difference vs MLP.
2. LSTM solves vanishing gradient via cell state Cₜ (memory highway) + 3 gates: forget (fₜ), input (iₜ), output (oₜ). Output shapes: output=(batch,seq,hidden); hn/cn=(num_layers,batch,hidden).
3. RNN limitation: sequential dependencies prevent parallelization; vanishing/exploding gradients; limited context window.
4. Static embeddings (Word2Vec) give ONE vector per word regardless of context (polysemy problem: "bank" financial vs river). Contextual embeddings (ELMo, BERT) solve this with bidirectional LSTM.
5. Loss function choice: MSE for regression (continuous output); Binary Cross-Entropy for 2-class (sigmoid); Categorical Cross-Entropy for 3+ classes (softmax).

## Exam-relevant concepts

### Loss Functions
- **Definition:** Measures how far predictions are from true labels. Optimizer uses loss to update weights via backprop.
- **Formula / numbers:**
  - **MSE (regression):** MSE = (1/n) × Σ(yᵢ − ŷᵢ)². Penalizes large errors heavily (squared). Perfect fit → MSE=0.
  - **Binary Cross-Entropy (2-class):** BCE = −(1/n) × Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]. Output layer: Sigmoid (0–1). Example: spam/not spam.
  - **Categorical Cross-Entropy (multi-class):** CCE = −(1/n) × Σᵢ Σc yᵢc · log(ŷᵢc). Output layer: Softmax (probs sum to 1). Example: cat/dog/bird.
- **When it matters (exam trap):**
  - MSE for regression (house price, temperature).
  - BCE for binary classification (spam detection, fraud, disease).
  - CCE for 3+ mutually exclusive classes (MNIST digits, image classification).
- **Common confusion:** Using MSE for classification (wrong; use cross-entropy). Forgetting that only the true-class probability matters in CCE.

### Parameters vs Hyperparameters
- **Definition:**
  - **Parameters:** Learned during training (weights, biases). Updated by optimizer. Set by: gradient descent.
  - **Hyperparameters:** Set before training (learning rate, batch size, epochs, # layers, dropout rate). Chosen by: data scientist.
- **When it matters:** "Is learning rate a parameter?" → NO (hyperparameter). "Are CNN filter weights parameters?" → YES.

### Backpropagation Weight Update Example
- **Setup (slide 9–18):** 2-layer net, x=1.0, w₁=0.6, w₂=0.4, w₃=0.8, w₄=0.5, target y=1.0, learning rate η=0.5.
- **Forward pass:**
  - h₁ = σ(w₁·x) = σ(0.6) = 0.6457.
  - h₂ = σ(w₂·x) = σ(0.4) = 0.5987.
  - ŷ = σ(w₃·h₁ + w₄·h₂) = σ(0.8159) = 0.6934.
  - Loss L = ½(y − ŷ)² = ½(1.0 − 0.6934)² = 0.0470.
- **Backward pass:**
  - σ'(z₃) = ŷ(1 − ŷ) = 0.6934 × 0.3066 = 0.2126.
  - δ_out = −(y − ŷ) × σ'(z₃) = −0.3066 × 0.2126 = −0.0652.
  - w₃_new = w₃ − η × δ_out × h₁ = 0.8000 + 0.5 × 0.0652 × 0.6457 = 0.8210 (Δ+0.0210).
  - w₄_new = 0.5195 (Δ+0.0195).
- **After 1 epoch:** Loss 0.0470 → 0.0454 (−3.5% drop). ŷ improves 0.6934 → 0.6987.
- **When it matters:** Demonstrates chain rule in action. Larger hidden state h₁ → larger weight update Δw₃.

### Underfitting, Good Fit, Overfitting
- **Definition:**
  - **Underfitting:** Model too simple → fails to capture patterns → high train error, high test error. Cause: too few layers/features, undertraining. Fix: more capacity, train longer.
  - **Good Fit:** Learns patterns, generalizes → low train error, low test error (small gap). Goal state.
  - **Overfitting:** Model memorizes noise → very low train error, high test error (large gap). Cause: too complex model, too little data. Fix: L2/L1 regularization, dropout, more data, early stopping.
- **When it matters (exam trap):**
  - Train loss ↓ but Val loss ↑ → **overfitting**.
  - Both losses stay high → **underfitting**.
- **Common confusion:** Thinking low training error = success (ignores generalization).

### Bias-Variance Tradeoff
- **Definition:**
  - **High Bias = Underfitting:** Model assumptions too rigid (e.g., straight line on curved data). High train & test error.
  - **High Variance = Overfitting:** Model too sensitive to training data (captures noise). Low train, high test error.
  - **Sweet Spot = Good Fit:** Tune hyperparameters (# layers, regularization) to balance. Monitor via train/val loss gap.
- **When it matters:** Informs model selection and regularization tuning.

### Regularization Techniques Recap
- **L1 (Lasso):** Loss + λ·Σ|w| → sparse model (some weights → 0).
- **L2 (Ridge):** Loss + λ·Σw² → shrinks all weights uniformly.
- **Dropout:** Randomly zero neurons during training (rate p). Prevents co-adaptation.
- **Early Stopping:** Monitor val loss; stop when it stops improving (patience window).
- **When it matters:** If question asks "how to fix overfitting," answers include all of the above + more data.

### Convolutional Neural Networks (CNNs)
- **Definition:** Specialized deep learning architecture for grid-structured data (images). Inspired by visual cortex. Uses learnable filters (kernels) that slide over input to detect spatial patterns (edges, textures, shapes). Achieves translation invariance (feature detected anywhere in image).
- **Key properties:**
  - **Local Connectivity:** Neurons connect to small regions (receptive field), NOT entire input. This is the STRUCTURAL difference vs MLP.
  - **Parameter Sharing:** Same filter applied across all spatial locations → fewer parameters than fully connected.
  - **Spatial Hierarchy:** Low-level (edges) → Mid-level (textures, shapes) → High-level (object parts) via stacked layers.
- **When it matters (exam trap):** MCQ asks "CNN vs MLP structural difference" → answer is LOCAL CONNECTIVITY (not activation function, not channels, not matrix mult).
- **Common confusion:** Thinking convolution is about the activation function (ReLU). No — it's about the sliding filter operation (dot product over spatial window).

### CNN Architecture Flow
- **Pipeline:** Input Image → Conv Layer → ReLU → Pooling → Conv Layer → Flatten → Fully Connected → Softmax → Output.
- **Dimensions:**
  - **Feature map depth increases** (more filters → more channels).
  - **Spatial size decreases** (pooling downsamples).
  - Early layers: edges & simple patterns. Mid layers: textures, shapes. Deep layers: high-level semantic concepts.
- **When it matters:** Understanding why spatial dimensions shrink (pooling) and why depth grows (more filters).

### CNN Key Components
- **Convolutional Layer:**
  - Applies learnable filters (kernels) via dot product.
  - Parameters: kernel size (e.g., 3×3), stride, padding, # filters.
  - Produces feature maps capturing spatial patterns.
- **Activation (ReLU):**
  - f(x) = max(0, x). Introduces non-linearity. Prevents vanishing gradient.
  - Variants: Leaky ReLU, ELU, GELU.
- **Pooling Layer:**
  - Reduces spatial dimensions (downsampling). Max pooling takes max value in region (e.g., 2×2 window).
  - Provides translation invariance, reduces parameters.
- **Fully Connected Layer:**
  - Flattens spatial features into vector → performs final classification/regression.
  - Often uses Dropout for regularization.

### Convolution Matrix Operation (Slide 29)
- **Input:** 5×5 image.
- **Filter/kernel:** 3×3.
- **Output:** 3×3 feature map (stride=1, no padding).
- **Dot product example (top-left patch):**
  - Patch: [[1,2,3],[0,1,2],[1,3,2]]. Filter: [[-1,0,1],[-2,0,2],[-1,0,1]].
  - Dot product: (1×-1)+(2×0)+(3×1)+(0×-2)+(1×0)+(2×2)+(1×-1)+(3×0)+(2×1) = 7.
- **When it matters:** Demonstrates how filter slides across every position (stride=1) → builds full feature map.

### Pooling Example (Slide 30)
- **Input:** 4×4 feature map.
- **Max Pool (2×2, stride=2):** Each 2×2 window → one output value (max).
- **Output:** 2×2 pooled map. Example: top-left window [12,20,18,36] → max=36.
- **When it matters:** Shows how pooling halves spatial dimensions (4×4 → 2×2).

### Fully Connected Layer Example (Slide 31)
- **Step 1 — Flatten:** 2×2 pooled map [[36,22],[30,25]] → vector [36,22,30,25] (4×1).
- **Step 2 — Multiply by weight matrix W (3×4):** W·x + b → scores [19.5, −1.8, 11.5] for 3 classes.
- **Step 3 — Softmax:** Scores → probabilities [100%, 0%, 0%] (Cat wins).
- **When it matters:** Demonstrates FC layer learns which feature combinations predict each class.

### CNN Landmarks
- **LeNet (1989):** Digit recognition (LeCun). First successful CNN.
- **AlexNet (2012):** Revolutionized ImageNet; proved deep learning at scale. First use of ReLU, Dropout, GPU training.
- **VGGNet (2014):** Very deep (16–19 layers), small 3×3 filters throughout.
- **ResNet (2015):** Residual connections (skip connections) enable training 50–152 layers without vanishing gradient.
- **EfficientNet (2019):** Compound scaling (depth, width, resolution) for optimal accuracy/efficiency tradeoff.

### CNN Applications
- **Computer Vision:** Image classification, object detection (YOLO), semantic segmentation, facial recognition.
- **Medical Imaging:** Tumor detection (MRI), diabetic retinopathy, histopathology, COVID-19 CT screening.
- **Other Domains:** NLP (Text-CNN for sentence classification), audio classification, video understanding, autonomous driving.

### CNN Drawbacks (Slide 36)
- **Fixed-size input/output:** CNNs struggle with variable-length sequences (e.g., sentences of different lengths, videos of varying duration).
- **No temporal relationship:** CNNs process data points independently or within a small local window. Cannot capture order/context across time steps → inefficient for tasks where order matters (stock forecasting, text generation).
- **Why this matters:** Motivates Recurrent Neural Networks (RNNs).

### Recurrent Neural Networks (RNNs)
- **Definition:** Networks with memory (hidden state hₜ) that carries information across time steps. Designed for sequential/temporal data where order matters.
- **Update rule (Vanilla RNN):** hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b).
- **Applications:** Machine translation, sentiment analysis, named entity recognition, text generation, speech recognition, music generation, stock forecasting.
- **When it matters:** If data is sequential (text, audio, time series, DNA, video), use RNN.

### RNN Mathematics
- **Hidden state update:** hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b).
- **Output:** yₜ = g(Wᵧ·hₜ) where g is activation (e.g., softmax for classification).
- **Loss (Cross-Entropy):** L = −Σₜ log P(yₜ | x₁…xₜ).
- **Parameters:** Wₕ (hidden-to-hidden), Wₓ (input-to-hidden), Wᵧ (hidden-to-output), b (bias).

### RNN Training: Backprop Through Time (BPTT)
- **Procedure:**
  1. Unfold RNN across T time steps.
  2. Compute gradients at each step.
  3. Sum gradients and update weights.
  4. Gradient = chain of partial derivatives across time.
- **Vanishing Gradient:** When |dhₜ/dhₜ₋₁| < 1, product shrinks exponentially → gradients → 0 → network stops learning long-range dependencies.
- **Exploding Gradient:** When |dhₜ/dhₜ₋₁| > 1, product grows exponentially → NaN weights. Fix: gradient clipping (clip at threshold).
- **Solution:** LSTM (Long Short-Term Memory).

### LSTM (Long Short-Term Memory)
- **Definition:** Solves vanishing gradient via cell state Cₜ (memory highway) that runs straight through time with minimal modification. Uses 3 gates to regulate information flow.
- **Key Innovation:** Dedicated cell state Cₜ (long-term memory) separate from hidden state hₜ (short-term output).
- **Gates (all use sigmoid σ to output 0–1):**
  1. **Forget Gate (fₜ):** Decides what % of old cell state to erase. fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf). 0=forget all, 1=keep all.
  2. **Input Gate (iₜ):** Decides which new values to write. iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi).
  3. **Cell Candidate (C̃ₜ):** New info to add. C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc).
  4. **Cell State Update:** Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ (⊙ = element-wise multiply).
  5. **Output Gate (oₜ):** Filters cell state → hidden. oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo).
  6. **Hidden State:** hₜ = oₜ ⊙ tanh(Cₜ).
- **When it matters (exam trap):**
  - LSTM has 2 states: cell state Cₜ (long-term) and hidden state hₜ (short-term).
  - Forget gate fₜ → discards old info. Input gate iₜ → adds new info. Output gate oₜ → filters output.
- **Common confusion:** Mixing up which gate does what. Remember: f=forget, i=input, o=output.

### LSTM Worked Example (Slide 47)
- **Setup:** hₜ₋₁=0.5, xₜ=1.0, Cₜ₋₁=0.3. All weights=0.5, biases=0.
- **Forget gate:** fₜ = σ(0.5×[0.5+1.0]) = σ(0.75) ≈ 0.679 (keep 67.9% of old cell).
- **Input gate:** iₜ = σ(0.75) ≈ 0.679.
- **Cell candidate:** C̃ₜ = tanh(0.75) ≈ 0.635.
- **Cell state update:** Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ = 0.679×0.3 + 0.679×0.635 = 0.204 + 0.431 = 0.635.
- **Output gate:** oₜ = σ(0.75) ≈ 0.679.
- **Hidden state:** hₜ = oₜ⊙tanh(Cₜ) = 0.679×tanh(0.635) = 0.679×0.561 ≈ 0.381.
- **When it matters:** Shows how gates combine to update memory (Cₜ) and output (hₜ).

### LSTM Output Shapes (PyTorch/TensorFlow)
- **Common trap:** LSTM returns 3 things:
  1. **output:** (batch, seq_len, hidden_size) — hidden states at each time step.
  2. **hn:** (num_layers, batch, hidden_size) — final hidden state.
  3. **cn:** (num_layers, batch, hidden_size) — final cell state.
- **When it matters (exam trap):** MCQ asks "LSTM output shape for batch=32, seq=10, hidden=128?" → output=(32,10,128); hn=(num_layers,32,128); cn=(num_layers,32,128).
- **Common confusion:** Mixing up seq_len with batch (they're different dimensions).

### CNN vs RNN Summary
| Aspect | CNN | RNN |
|--------|-----|-----|
| **Best For** | Spatial/grid data (images) | Sequential/temporal data (text, audio, time series) |
| **Memory** | None (feedforward) | Hidden state hₜ across timesteps |
| **Key Operation** | Convolution + pooling | Recurrent weight updates |
| **Parallelization** | Easy (independent spatial patches) | Hard (sequential dependencies) |
| **Famous Architectures** | ResNet, EfficientNet, VGG | LSTM, GRU, Transformer |

### RNN Limitations (Slide 49)
1. **Slow computation for long sequences:** Cannot parallelize due to timestep dependencies (hₜ depends on hₜ₋₁).
2. **Vanishing/Exploding gradient:** Limits effective context window (LSTM helps but doesn't fully solve).
3. **Limited information in hidden state:** History is "forgotten" after many timesteps (compression bottleneck).
4. **Long-range dependencies hard:** Distant context (100+ tokens ago) is lost.
- **When it matters:** Motivates Transformer architecture (next evolution: self-attention, parallelizable).

### Evolution of Language Processing (Slide 48)
- **Statistical NLP:** Probabilistic LM, HMM, CRF. Feature extraction by hand.
- **Deep NLP:** Word2Vec, CNN, RNN/LSTM/GRU. Task-specific, supervised.
- **Transformers:** Self-attention. Parallelizable. BERT (encoder-only), GPT (decoder-only).
- **Pre-Training + Fine-Tuning:** Pre-train once (e.g., BERT on Wikipedia), fine-tune on multiple tasks (classification, NER, QA).
- **Foundation Models:** GPT-X, Llama. Prompt engineering instead of fine-tuning.
- **When it matters:** Context for "why Transformers now?" (solves RNN parallelization, long-range dependency issues).

### Contextual Embeddings (ELMo)
- **Problem:** Word2Vec gives ONE static vector per word (polysemy: "bank" financial vs river cannot be distinguished).
- **Solution:** ELMo (Embeddings from Language Models) uses bidirectional LSTM → forward pass (left-to-right) + backward pass (right-to-left) → "bank" gets context from BOTH directions before producing vector.
- **Result:** Same word, different contexts → different vectors.
- **Example (Slide 67):**
  - Sentence 1: "I deposited money at the **bank**" → bank_finance vector.
  - Sentence 2: "She sat on the river **bank**" → bank_river vector.
  - Cosine similarity ≈ 0.54 (different meanings → different vectors).
- **When it matters (exam trap):** If MCQ asks "how does ELMo handle polysemy?" → bidirectional LSTM computes context-dependent embeddings.

### One-Hot vs Word2Vec vs ELMo Summary
| Representation | Dimension | Semantic Similarity | Context-Aware |
|----------------|-----------|---------------------|---------------|
| **One-Hot** | Vocab size (50k) | NO (dot product=0) | NO (static) |
| **Word2Vec** | ~300 | YES (cosine ~0.92 for cat/kitten) | NO (static: one vector per word) |
| **ELMo** | ~1024 (3 layers) | YES | YES (bidirectional LSTM → different vectors per context) |

## Diagrams / architectures described

### CNN Architecture (Slide 27)
- **Pipeline:** Input Image (224×224×3) → Conv Layer (learns local features) → ReLU (non-linearity) → Pooling (reduces spatial dims) → Conv Layer (higher-level features) → Flatten (spatial → vector) → Fully Connected (class scores) → Softmax (probabilities).
- **Dimensions:** Feature map depth increases (more filters) → spatial size decreases (pooling).

### Convolution Operation (Slide 29)
- **Input:** 5×5 matrix.
- **Filter:** 3×3 kernel (e.g., edge detector [[-1,0,1],[-2,0,2],[-1,0,1]]).
- **Output:** 3×3 feature map (stride=1). Each value = dot product of filter with corresponding 3×3 patch.

### Max Pooling (Slide 30)
- **Input:** 4×4 feature map.
- **Operation:** 2×2 window, stride=2 → take max value per window.
- **Output:** 2×2 (halved spatial dimensions).

### Fully Connected Layer (Slide 31)
- **Step 1:** Flatten 2×2 pooled map → [36,22,30,25] (4×1 vector).
- **Step 2:** W(3×4) · x(4×1) + b → scores [19.5, −1.8, 11.5].
- **Step 3:** Softmax → [1.0, 0.0, 0.0] (Cat=100%).

### Vanilla RNN (Slide 39)
- **Structure:** RNN cell at each timestep. Hidden state hₜ flows right (memory).
- **Update rule:** hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b).
- **No gates:** No forget/input/output gates (unlike LSTM). Just hₜ carrying memory.

### LSTM Cell (Slides 42–46)
- **Cell state Cₜ:** Runs horizontally through time (conveyor belt). Modified only by forget gate (fₜ ⊙ Cₜ₋₁) and input gate (iₜ ⊙ C̃ₜ).
- **Forget gate fₜ:** Sigmoid (0=forget, 1=keep). Controls what % of Cₜ₋₁ to discard.
- **Input gate iₜ:** Sigmoid. Controls what % of new candidate C̃ₜ to write.
- **Cell candidate C̃ₜ:** Tanh (−1 to 1). New info to potentially add.
- **Cell state update:** Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ.
- **Output gate oₜ:** Sigmoid. Filters cell state → hidden state.
- **Hidden state hₜ:** hₜ = oₜ⊙tanh(Cₜ). Passed to next timestep and as output.

### BPTT (Backprop Through Time)
- **Forward pass:** Unfold RNN across T steps → compute outputs y₁, y₂, …, yₜ.
- **Loss:** L = Σₜ loss(yₜ, target_t).
- **Backward pass:** Gradients flow backward through time (chain rule across T steps). Sum gradients → update shared weights Wₕ, Wₓ, Wᵧ.
- **Problem:** Gradient can vanish (product of T terms <1) or explode (product >1).

### ELMo BiLSTM (Slide 67)
- **Forward LSTM:** Processes sentence left-to-right (I → went → to → the → bank).
- **Backward LSTM:** Processes sentence right-to-left (bank → the → to → went → I).
- **Concatenation:** For each word, combine forward and backward hidden states → context-aware embedding.
- **Result:** "bank" in "I deposited money at the bank" gets different vector than "bank" in "She sat on the river bank."

### Evolution Timeline (Slide 48)
- **Statistical NLP:** HMM, CRF (1990s–2000s).
- **Deep NLP:** Word2Vec (2013), CNN for text, RNN/LSTM/GRU (2010s).
- **Transformers:** Self-attention (2017). BERT (encoder-only, 2018), GPT (decoder-only, 2018).
- **Foundation Models:** GPT-3 (2020), GPT-4, Llama (2020s). Prompt engineering.

## Likely MCQ angles
1. **Loss function selection:** "Scenario: predicting house price (continuous output)" → MSE. "Scenario: spam/not spam (binary)" → Binary Cross-Entropy. "Scenario: classify MNIST digits (10 classes)" → Categorical Cross-Entropy.
2. **CNN vs MLP structural difference:** "What is the key architectural difference between CNN and MLP?" → LOCAL CONNECTIVITY (CNN neurons connect to small receptive field; MLP neurons connect to entire input).
3. **Convolution output size:** "Input 5×5, filter 3×3, stride=1, no padding → output size?" → 3×3. Formula: (W−F)/S + 1 = (5−3)/1 + 1 = 3.
4. **Pooling effect:** "Max pool 2×2, stride=2 on 4×4 feature map → output size?" → 2×2 (halves dimensions).
5. **LSTM output shapes:** "LSTM with batch=32, seq_len=10, hidden_size=128, num_layers=2 → output.shape, hn.shape, cn.shape?" → output=(32,10,128); hn=(2,32,128); cn=(2,32,128).
6. **LSTM gates:** "Which gate decides what to forget from cell state?" → Forget gate (fₜ). "Which gate decides what new info to write?" → Input gate (iₜ). "Which gate filters output?" → Output gate (oₜ).
7. **RNN vanishing gradient:** "Why do vanilla RNNs struggle with long sequences?" → Vanishing gradient (product of T derivatives <1 → gradient → 0). "Solution?" → LSTM (cell state highway + gates).
8. **CNN vs RNN use case:** "Task: image classification" → CNN. "Task: machine translation" → RNN/LSTM or Transformer. "Task: speech recognition" → RNN/LSTM (sequential audio).
9. **Word2Vec limitation:** "Why can't Word2Vec handle polysemy (bank financial vs river)?" → Gives ONE static vector per word type regardless of context. "Solution?" → ELMo (bidirectional LSTM → context-dependent embeddings).
10. **ELMo mechanism:** "How does ELMo produce context-aware embeddings?" → Bidirectional LSTM (forward + backward passes) → combines left and right context before producing vector.
11. **Parameter vs hyperparameter:** "Is dropout rate a parameter?" → NO (hyperparameter). "Are CNN filter weights parameters?" → YES (learned).
12. **Regularization for overfitting:** "Train loss 0.02, Val loss 0.18 — what's wrong and how to fix?" → Overfitting. Fix: L2 penalty, dropout, early stopping, more data.
13. **Backprop weight update direction:** "In gradient descent, weights update as w ← w − η·∂L/∂w. Why minus?" → Move opposite to gradient (downhill toward loss minimum).
14. **CNN landmarks:** "Which CNN introduced residual connections (skip connections)?" → ResNet (2015). "Which CNN first used ReLU, Dropout, GPU training?" → AlexNet (2012).
15. **RNN limitation:** "Why are RNNs slow on long sequences?" → Sequential dependencies (hₜ depends on hₜ₋₁) → cannot parallelize. "Why Transformers solve this?" → Self-attention is parallelizable.

## One-line flashcards
- Q: MSE formula → A: (1/n) × Σ(yᵢ − ŷᵢ)²
- Q: Binary Cross-Entropy formula → A: −(1/n) × Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
- Q: Categorical Cross-Entropy formula → A: −(1/n) × Σᵢ Σc yᵢc · log(ŷᵢc)
- Q: When to use MSE → A: Regression (continuous output)
- Q: When to use Binary Cross-Entropy → A: 2-class classification (spam/not spam)
- Q: When to use Categorical Cross-Entropy → A: 3+ mutually exclusive classes (cat/dog/bird)
- Q: Parameter definition → A: Learned during training (weights, biases)
- Q: Hyperparameter definition → A: Set before training (learning rate, # layers, dropout rate)
- Q: CNN local connectivity → A: Neurons connect to small receptive field, not entire input (vs MLP)
- Q: CNN parameter sharing → A: Same filter reused across all spatial locations
- Q: Convolution output size formula → A: (W−F)/S + 1
- Q: Max pooling 2×2 stride=2 effect → A: Halves spatial dimensions (4×4 → 2×2)
- Q: ReLU formula → A: f(x) = max(0, x)
- Q: Sigmoid formula → A: σ(z) = 1 / (1 + e^−z)
- Q: Softmax output property → A: All probabilities sum to 1
- Q: CNN landmark: LeNet → A: 1989, digit recognition (LeCun)
- Q: CNN landmark: AlexNet → A: 2012, ImageNet revolution, first ReLU/Dropout/GPU
- Q: CNN landmark: ResNet → A: 2015, residual connections (skip connections)
- Q: CNN drawback → A: Fixed-size input/output; no temporal relationships
- Q: RNN definition → A: Network with hidden state hₜ (memory) across time steps
- Q: RNN update rule → A: hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)
- Q: BPTT definition → A: Backprop Through Time (unfold RNN, compute gradients across T steps)
- Q: Vanishing gradient cause (RNN) → A: |dhₜ/dhₜ₋₁| < 1 → product shrinks exponentially
- Q: Exploding gradient fix → A: Gradient clipping (clip at threshold)
- Q: LSTM key innovation → A: Cell state Cₜ (memory highway) + 3 gates
- Q: LSTM forget gate fₜ → A: Decides what % of old cell state to erase (0=forget all, 1=keep all)
- Q: LSTM input gate iₜ → A: Decides what % of new info to write
- Q: LSTM output gate oₜ → A: Filters cell state → hidden state
- Q: LSTM cell state update → A: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ
- Q: LSTM hidden state formula → A: hₜ = oₜ⊙tanh(Cₜ)
- Q: LSTM output shape → A: (batch, seq_len, hidden_size)
- Q: LSTM hn shape → A: (num_layers, batch, hidden_size)
- Q: LSTM cn shape → A: (num_layers, batch, hidden_size)
- Q: LSTM vs vanilla RNN → A: LSTM has cell state + gates; vanilla RNN just has hₜ
- Q: CNN vs RNN: spatial data → A: CNN (images)
- Q: CNN vs RNN: sequential data → A: RNN/LSTM (text, audio, time series)
- Q: RNN limitation 1 → A: Sequential dependencies → cannot parallelize
- Q: RNN limitation 2 → A: Vanishing/exploding gradient → limited context window
- Q: RNN limitation 3 → A: History compressed in fixed-size hₜ → "forgotten" after many steps
- Q: Word2Vec polysemy problem → A: ONE static vector per word (can't distinguish "bank" financial vs river)
- Q: ELMo solution → A: Bidirectional LSTM → context-aware embeddings
- Q: ELMo mechanism → A: Forward pass (left-to-right) + backward pass (right-to-left) → combine contexts
- Q: One-hot vector dimension → A: Vocabulary size (e.g., 50k)
- Q: Word2Vec embedding dimension → A: ~300 (fixed, dense)
- Q: ELMo embedding dimension → A: ~1024 (3 LSTM layers)
- Q: One-hot semantic similarity → A: 0 (orthogonal by construction)
- Q: Word2Vec king − man + woman → A: ≈ queen (analogy learned from co-occurrence)
- Q: Word2Vec Skip-gram → A: Center word → predict context
- Q: Word2Vec CBOW → A: Context → predict center word
- Q: Evolution: Statistical NLP → A: HMM, CRF, hand-crafted features (1990s–2000s)
- Q: Evolution: Deep NLP → A: Word2Vec, CNN, RNN/LSTM (2010s)
- Q: Evolution: Transformers → A: Self-attention, BERT, GPT (2017+)
- Q: Evolution: Foundation Models → A: GPT-3/4, Llama, prompt engineering (2020s)
- Q: Underfitting symptom → A: High train error, high test error
- Q: Overfitting symptom → A: Low train error, high test error (large gap)
- Q: Overfitting fix → A: L2/L1 regularization, dropout, early stopping, more data
- Q: Bias-Variance: High Bias → A: Underfitting (model too simple)
- Q: Bias-Variance: High Variance → A: Overfitting (model too complex)
- Q: Dropout mechanism → A: Randomly zero neurons during training (rate p)
- Q: Early stopping trigger → A: Val loss stops improving for patience epochs
- Q: L1 regularization effect → A: Sparse model (some weights → 0)
- Q: L2 regularization effect → A: Shrinks all weights uniformly
- Q: Backprop weight update → A: w ← w − η·∂L/∂w (move opposite to gradient)
- Q: Learning rate η → A: Hyperparameter controlling step size in gradient descent
- Q: Batch size → A: Hyperparameter: # samples per gradient update
- Q: Epochs → A: Hyperparameter: # full passes through training data
- Q: CNN fully connected layer → A: Flattens spatial features → performs classification
- Q: CNN flatten operation → A: 2D/3D feature map → 1D vector
- Q: Softmax output layer → A: Multi-class classification (3+ classes)
- Q: Sigmoid output layer → A: Binary classification (2 classes)
- Q: Linear output layer → A: Regression (continuous value)
- Q: PyTorch nn.Conv2d params → A: in_channels, out_channels, kernel_size, padding, stride
- Q: PyTorch nn.MaxPool2d params → A: kernel_size, stride
- Q: PyTorch nn.Linear params → A: in_features, out_features
- Q: PyTorch nn.Dropout param → A: p (probability of zeroing a neuron)
- Q: Gradient clipping purpose → A: Prevent exploding gradients (RNN training)
- Q: Tanh output range → A: (−1, 1)
- Q: ReLU output range → A: [0, ∞)
- Q: Sigmoid output range → A: (0, 1)
- Q: LSTM gates use which activation → A: Sigmoid (for gates fₜ, iₜ, oₜ); tanh (for cell candidate C̃ₜ and cell state filtering)
- Q: Element-wise multiply notation → A: ⊙ (Hadamard product)
- Q: LSTM conveyor belt → A: Cell state Cₜ (memory highway)
- Q: LSTM vs GRU → A: GRU has 2 gates (reset, update); LSTM has 3 gates (forget, input, output)
- Q: Transformer motivation → A: Solve RNN parallelization issue + long-range dependencies via self-attention

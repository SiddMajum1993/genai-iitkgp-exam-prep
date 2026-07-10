# Lecture 1 — Introduction to Artificial Intelligence: Machine Learning and Deep Learning
**Instructor:** Prof. Niloy Ganguly, IIT Kharagpur | **Module:** Foundations

## TL;DR (5 bullets max)
1. Confusion matrix metrics (Precision, Recall, F1) drive ML evaluation; Recall for life-threatening tasks, Precision when false alarms are costly.
2. Sparse one-hot vectors cannot capture semantic similarity (dot product always = 0); Word2Vec learns dense embeddings from co-occurrence.
3. Underfitting (high bias) vs Overfitting (high variance): regularization (L1/L2), dropout, early stopping control complexity.
4. MLPs learn through backpropagation: forward pass → compute loss → backward pass (chain rule) → update weights with gradient descent.
5. Embeddings enable king − man + woman = queen via distributional hypothesis (Firth, 1957): "You shall know a word by the company it keeps."

## Exam-relevant concepts

### Machine Learning Definition (Tom Mitchell, 1997)
- **Definition:** A program learns from experience E with respect to task T and performance measure P if performance on T improves with E.
- **When it matters:** Identifies the three components of any ML system: task, data, evaluation metric.
- **Common confusion:** Not just "AI that predicts" — must have explicit task/metric/data triad.

### Confusion Matrix
- **Definition:** 2×2 table comparing predicted class vs actual class: TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative).
- **Formula / numbers:**
  - **Accuracy:** (TP + TN) / Total
  - **Precision:** TP / (TP + FP) — "Of all positive predictions, how many were correct?"
  - **Recall:** TP / (TP + FN) — "Of all actual positives, how many did we catch?"
  - **F1 Score:** 2 × (P × R) / (P + R) — harmonic mean
- **When it matters (exam trap):**
  - **High Recall needed:** Disease screening, cancer detection (missing a positive is dangerous).
  - **High Precision needed:** Spam filter, fraud detection (false alarms are costly).
  - Accuracy is misleading on imbalanced datasets (e.g., 95% non-fraud → always predict "not fraud" = 95% accuracy but useless).
- **Common confusion:** Mixing up FP and FN; forgetting that Precision cares about "positive predictions" while Recall cares about "actual positives."

### Supervised vs Unsupervised Learning
- **Definition:**
  - **Supervised:** Training data has input-output pairs (labels). Learn mapping f: X → Y. Examples: classification (cat/dog), regression (house price).
  - **Unsupervised:** No labels. Discover hidden structure. Examples: clustering (K-Means, DBSCAN), dimensionality reduction (PCA, t-SNE).
- **When it matters:** If the question involves labeled data, it's supervised; if it's about grouping or compression, it's unsupervised.

### Train/Test Split
- **Definition:** Divide dataset into training set (to learn) and test set (to evaluate). Standard split: 80-20.
- **Why:** Prevents overfitting. Model never sees test data during training → measures generalization.
- **Common confusion:** Using test set for hyperparameter tuning contaminates evaluation (should use validation set).

### Feature Scaling (StandardScaler)
- **Definition:** Normalize features so they have similar ranges. Improves convergence for gradient-based methods.
- **Formula / numbers:**
  - **Standardization (Z-score):** x' = (x - μ) / σ (mean = 0, std = 1)
  - **Min-Max scaling:** x' = (x - min) / (max - min) (range [0,1])
- **When it matters:** Logistic regression, SVM, neural nets are scale-sensitive. Tree-based models (Random Forest, XGBoost) are not.

### Sparse vs Dense Representations
- **Definition:**
  - **One-hot (sparse):** Each word = |V|-dimensional vector with one 1, rest 0. No semantic meaning (dot product always 0).
  - **Embeddings (dense):** Fixed-length (e.g., 300-dim) vector where geometry = meaning. Similar words cluster.
- **When it matters (exam trap):**
  - One-hot: 50k vocab → 50k dims, 99%+ zeros, no similarity (cat · kitten = 0).
  - Word2Vec: 50k vocab → 300 dims, dense floats, cat · kitten ≈ 0.92.
  - King − man + woman ≈ queen emerges from distributional learning, NOT from one-hot.
- **Common confusion:** Thinking one-hot can capture analogy relationships (it cannot).

### Distributional Hypothesis (Firth, 1957)
- **Definition:** "You shall know a word by the company it keeps." A word's meaning is defined by the words that frequently appear in its context window (±k words).
- **When it matters:** Explains how Word2Vec learns embeddings: similar contexts → similar vectors.
- **Formula:** Context window size k (e.g., k=2 means ±2 words around target).

### Word Embeddings (Word2Vec)
- **Definition:** Dense vector representations learned from co-occurrence patterns. Two training strategies:
  - **Skip-gram:** Center word → predict context words. Better for rare words.
  - **CBOW:** Context words → predict center word. Faster, better for frequent words.
- **Formula / numbers:**
  - Typical embedding dimension: 50–300 (vs 50,000 for one-hot).
  - Famous analogy: vec(king) − vec(man) + vec(woman) ≈ vec(queen) with cosine similarity ~0.99.
- **When it matters (exam trap):** Word2Vec gives ONE static vector per word regardless of context (polysemy problem: "bank" financial vs river cannot be distinguished).
- **Common confusion:** Thinking Word2Vec uses hand-crafted rules (it learns from unsupervised co-occurrence prediction).

### Perceptron and MLP (Multi-Layer Perceptron)
- **Definition:**
  - **Perceptron (Rosenblatt, 1958):** Single-layer model: z = Σ wᵢxᵢ + b; ŷ = f(z). Can only learn linearly separable problems.
  - **MLP:** Multi-layer network with hidden layers. Each connection has a learnable weight. Activation functions (sigmoid, ReLU) add non-linearity.
- **When it matters:** MLP enables learning non-linear decision boundaries via hidden layers; perceptron alone cannot.

### Activation Functions
- **Definition:** Introduce non-linearity to enable complex mappings.
  - **Sigmoid:** σ(z) = 1 / (1 + e^−z). Output ∈ (0,1). Used for binary classification output.
  - **ReLU:** f(x) = max(0, x). Prevents vanishing gradients. Most common in hidden layers.
  - **Tanh:** tanh(z) = (e^z − e^−z) / (e^z + e^−z). Output ∈ (−1, 1). Zero-centered.
- **When it matters (exam trap):** ReLU solves vanishing gradient; sigmoid/tanh saturate at extremes (gradient → 0).

### Backpropagation and Gradient Descent
- **Definition:** Training algorithm for neural networks.
  1. **Forward pass:** Input flows through network → produces prediction ŷ.
  2. **Loss function:** Measures error (e.g., MSE for regression, cross-entropy for classification).
  3. **Backward pass:** Compute gradient of loss w.r.t. each weight using chain rule: ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w.
  4. **Update weights:** w ← w − η · ∂L/∂w (η = learning rate).
- **Formula / numbers:**
  - **MSE (regression):** L = (1/n)Σ(yᵢ − ŷᵢ)²
  - **Binary Cross-Entropy:** L = −(1/n)Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
  - **Categorical Cross-Entropy:** L = −(1/n)Σᵢ Σc yᵢc · log(ŷᵢc)
- **When it matters:** Chain rule enables learning in deep nets. Variants: SGD, Mini-batch GD, Adam, RMSProp.
- **Common confusion:** Vanishing gradients occur when derivatives shrink through many layers (especially with sigmoid/tanh).

### Underfitting vs Overfitting
- **Definition:**
  - **Underfitting (High Bias):** Model too simple → cannot capture patterns → high train & test error.
  - **Overfitting (High Variance):** Model too complex → memorizes noise → low train error, high test error.
  - **Good Fit:** Balances bias and variance → low train & test error.
- **When it matters (exam trap):**
  - Train loss ↓ but Val loss ↑ → **overfitting** → add regularization, dropout, early stopping.
  - Both losses high → **underfitting** → increase model capacity, train longer.
- **Common confusion:** Thinking high training accuracy always means success (ignores test performance).

### Bias-Variance Tradeoff
- **Formula:** Total Error = Bias² + Variance + irreducible noise (ε).
- **Definition:**
  - **High Bias:** Rigid model assumptions (e.g., linear model on curved data).
  - **High Variance:** Too sensitive to training data fluctuations.
- **When it matters:** Tuning hyperparameters (# layers, regularization strength) navigates this tradeoff.

### Regularization Techniques
- **Definition:** Add penalty to loss function to discourage complexity.
  - **L2 (Ridge):** Loss + λ·Σw² → shrinks all weights smoothly toward zero.
  - **L1 (Lasso):** Loss + λ·Σ|w| → drives some weights exactly to zero (feature selection).
  - **Dropout:** Randomly zero out neurons during training (rate p, e.g., p=0.5). Prevents co-adaptation.
  - **Early Stopping:** Monitor validation loss; stop training when it stops improving (patience window).
  - **Batch Normalization:** Normalize layer activations for stable training.
  - **Data Augmentation:** Expand training set with transforms (flip, crop, noise).
- **When it matters (exam trap):**
  - L1 creates sparse models (some weights = 0); L2 keeps all features but shrinks them.
  - Dropout is only active during training (turned off at inference).
- **Common confusion:** Confusing L1/L2 penalties (L1 = absolute value, L2 = squared).

### L1 vs L2 Geometry
- **Definition:**
  - L1 constraint: diamond shape → solution often hits axes → sparse weights.
  - L2 constraint: circle shape → solution shrinks uniformly → no exact zeros.
- **When it matters:** If question asks "which regularization for feature selection," answer L1 (Lasso).

### PCA (Principal Component Analysis)
- **Definition:** Unsupervised dimensionality reduction. Finds orthogonal axes (principal components) that maximize variance.
- **Why:** High-dimensional data (images: 224×224×3 = 150k dims; text: 100k vocab; genomics: 20k genes) suffers from curse of dimensionality.
- **When it matters:** Compress d=1000 dims → k=2 while keeping structure. No labels needed.
- **Common confusion:** PCA is linear; t-SNE is non-linear (better for visualization but slower).

### Parameters vs Hyperparameters
- **Definition:**
  - **Parameters:** Learned during training (weights, biases). Set by optimizer automatically.
  - **Hyperparameters:** Set before training (learning rate, batch size, # layers, dropout rate, epochs). Chosen by data scientist.
- **When it matters (exam trap):** "Is learning rate a parameter?" → NO, it's a hyperparameter.

### AI History Milestones
- **Definition:**
  - **Turing Test:** Can a machine fool a human into thinking it's human?
  - **1956 Dartmouth Conference:** Birth of AI.
  - **GOFAI (Good Old-Fashioned AI):** Logic, rules, semantic nets. Hit the wall due to ambiguity, scalability, brittleness.
  - **AI Winters:** 1970s (after failed translation systems), 1990s (after expert systems plateau).
  - **Boom 3 (2010s+):** Deep learning, big data, GPUs.
- **When it matters:** Context for "why deep learning now?" (data, compute, backprop at scale).

## Diagrams / architectures described

### MLP (Multi-Layer Perceptron)
- **Structure:** Input layer → Hidden layer(s) → Output layer. Each connection has weight wᵢⱼ.
- **Flow:** Forward: input → weighted sum → activation → next layer. Backward: gradients via chain rule.
- **Dimensions:** Hidden layer size is hyperparameter. Output layer: # classes for classification, 1 for regression.

### Backpropagation Computation Graph
- **Forward pass example (2-layer net, 1 input, 1 output):**
  - x=1.0, w₁=0.6, w₂=0.4, w₃=0.8, w₄=0.5, target y=1.0, η=0.5.
  - z₁ = w₁·x = 0.6 → h₁ = σ(z₁) = 0.6457.
  - z₂ = w₂·x = 0.4 → h₂ = σ(z₂) = 0.5987.
  - z₃ = w₃·h₁ + w₄·h₂ = 0.8159 → ŷ = σ(z₃) = 0.6934.
  - Loss L = ½(y − ŷ)² = 0.0470.
- **Backward pass:**
  - δ_out = −(y − ŷ) · σ'(z₃) = −0.3066 × 0.2126 = −0.0652.
  - w₃_new = w₃ − η · δ_out · h₁ = 0.8000 + 0.5 × 0.0652 × 0.6457 = 0.8210 (Δ +0.0210).
  - After 1 epoch: Loss drops from 0.0470 → 0.0454 (−3.5%).

### One-Hot vs Dense Embeddings
- **One-hot:** vocab size V → V-dimensional vector, one 1 at index i, rest 0. Example: "cat" = [1,0,0,0,0,0,0,0] (8-dim for 8-word vocab).
- **Dense:** vocab size V → d-dimensional vector (d << V, e.g., 300). Example: "cat" = [0.32, −0.71, 0.15, ..., 0.61] (300-dim).
- **Geometry:** Dense vectors allow cosine similarity: cos(cat, kitten) ≈ 0.92; one-hot: cos(cat, kitten) = 0.0.

### Distributional Semantics Context Window
- **Context window k=2:** For word "banking" in "government debt problems turning into **banking** crises as happened", context = [debt, problems, turning, into, crises, as].
- **Co-occurrence matrix:** Count how often each word appears in context of every other word → compress via SVD or train neural net (Word2Vec).

### Regularization Geometry (L1 vs L2)
- **L1 (diamond):** Constraint |w₁| + |w₂| ≤ t. Loss contours often intersect at axes → sparse solution (e.g., w = [0, 1.23]).
- **L2 (circle):** Constraint w₁² + w₂² ≤ t². Loss contours intersect smoothly → all weights shrink (e.g., w = [0.12, 0.88]).

### Underfitting / Good Fit / Overfitting Curves
- **Underfitting:** Model = straight line on curved data. Train error high, test error high.
- **Good Fit:** Model captures true pattern. Train error low, test error low (gap small).
- **Overfitting:** Model = high-degree polynomial through noise. Train error very low, test error high (large gap).

## Likely MCQ angles
1. **Confusion matrix arithmetic:** Given TP=40, FP=10, FN=5, TN=45, compute Precision, Recall, F1. Which metric matters for cancer screening? (Answer: Recall, because FN = missed cancer = dangerous.)
2. **When to use which metric:** Scenario: "Spam filter where marking real email as spam annoys users" → optimize for Precision (minimize FP). Scenario: "COVID screening where missing a case spreads disease" → optimize for Recall (minimize FN).
3. **Word2Vec analogy mechanism:** "How does king − man + woman = queen work?" → NOT rule-based; emerges from distributional similarity learned via co-occurrence prediction (Skip-gram or CBOW).
4. **One-hot vs embeddings:** "Why can't one-hot vectors capture similarity?" → Dot product of any two one-hot vectors = 0 (orthogonal by construction). "Why are embeddings better?" → Dense vectors where geometry encodes meaning.
5. **Regularization choice:** "Which regularization for feature selection?" → L1 (Lasso) drives weights to exact zero. "Which for smooth shrinkage?" → L2 (Ridge).
6. **Overfitting detection:** "Train loss 0.01, Val loss 0.15 — what's the problem?" → Overfitting (train << val). "Fix?" → L2 penalty, dropout, early stopping, more data.
7. **Backprop chain rule:** "In a 2-layer net, how is ∂L/∂w₁ computed?" → ∂L/∂w₁ = ∂L/∂ŷ · ∂ŷ/∂h₁ · ∂h₁/∂z₁ · ∂z₁/∂w₁ (chain rule layer by layer).
8. **Activation function choice:** "Which activation prevents vanishing gradient in deep nets?" → ReLU (gradient = 1 for x>0, not saturated like sigmoid/tanh).
9. **Parameters vs hyperparameters:** "Is the learning rate a parameter?" → NO (hyperparameter). "Is a weight a parameter?" → YES (learned).
10. **Distributional hypothesis:** "What does Firth's 'You shall know a word by the company it keeps' mean?" → Word meaning defined by context words in a window (distributional semantics).

## One-line flashcards
- Q: Precision formula → A: TP / (TP + FP)
- Q: Recall formula → A: TP / (TP + FN)
- Q: F1 formula → A: 2 × (P × R) / (P + R)
- Q: When to prioritize Recall → A: Disease screening, life-threatening tasks (missing positive = dangerous)
- Q: When to prioritize Precision → A: Spam/fraud detection (false alarms = costly)
- Q: Accuracy formula → A: (TP + TN) / Total
- Q: Accuracy limitation → A: Misleading on imbalanced datasets
- Q: StandardScaler formula → A: x' = (x − μ) / σ
- Q: One-hot vector dot product → A: Always 0 (orthogonal)
- Q: Word2Vec embedding dimension → A: ~300 (vs 50k for one-hot)
- Q: King − man + woman = ? → A: Queen (learned from co-occurrence)
- Q: Distributional hypothesis (Firth, 1957) → A: "You shall know a word by the company it keeps"
- Q: Word2Vec Skip-gram → A: Center word → predict context
- Q: Word2Vec CBOW → A: Context → predict center word
- Q: Word2Vec limitation → A: One static vector per word (polysemy problem)
- Q: MSE loss formula → A: L = (1/n)Σ(yᵢ − ŷᵢ)²
- Q: Binary Cross-Entropy formula → A: L = −(1/n)Σ[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
- Q: Backprop weight update → A: w ← w − η · ∂L/∂w
- Q: Underfitting symptoms → A: High train error, high test error
- Q: Overfitting symptoms → A: Low train error, high test error (large gap)
- Q: Overfitting fix → A: L2/L1 regularization, dropout, early stopping, more data
- Q: L1 regularization effect → A: Sparse model (some weights → 0)
- Q: L2 regularization effect → A: Shrinks all weights uniformly
- Q: Dropout mechanism → A: Randomly zero neurons during training (rate p)
- Q: Early stopping trigger → A: Val loss stops improving for patience epochs
- Q: ReLU formula → A: f(x) = max(0, x)
- Q: Sigmoid formula → A: σ(z) = 1 / (1 + e^−z)
- Q: Vanishing gradient cause → A: Derivatives shrink through layers (sigmoid/tanh saturate)
- Q: Parameter definition → A: Learned during training (weights, biases)
- Q: Hyperparameter definition → A: Set before training (learning rate, # layers, dropout rate)
- Q: PCA purpose → A: Dimensionality reduction via max variance axes
- Q: Supervised learning definition → A: Training data has input-output pairs (labels)
- Q: Unsupervised learning definition → A: No labels; discover structure (clustering, PCA)
- Q: Train/test split purpose → A: Evaluate generalization, prevent overfitting
- Q: Perceptron limitation → A: Can only learn linearly separable problems
- Q: MLP advantage → A: Hidden layers enable non-linear decision boundaries
- Q: Bias-Variance tradeoff → A: Total Error = Bias² + Variance + ε
- Q: High Bias meaning → A: Underfitting (model too simple)
- Q: High Variance meaning → A: Overfitting (model too complex)
- Q: Good Old AI failure → A: Brittleness, ambiguity, scalability issues (1970s AI Winter)

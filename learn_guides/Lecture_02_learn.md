# Lecture 2 — Loss Functions, CNNs, RNNs, and the Evolution to Contextual Embeddings
**Estimated read time:** 35–45 min · **Difficulty:** Foundation

## What this lecture is about

This lecture moves from shallow models (logistic regression, perceptron) to deep architectures designed for specific data types. You will learn why CNNs dominate image tasks (parameter sharing via convolution, spatial hierarchy via pooling), why RNNs handle sequences (hidden state as memory), and why LSTMs fix RNNs' vanishing gradient problem (gated cell state). You will also trace the evolution of word representations from static Word2Vec (one vector per word, polysemy problem) to contextual ELMo (bidirectional LSTM, different vectors for "bank" in finance vs. river contexts).

This matters for the exam because MCQs will test whether you understand *why* convolution reduces parameters compared to fully connected layers (answer: local connectivity + weight sharing), *when* to use MSE vs. cross-entropy (answer: regression vs. classification), and *how* LSTM gates solve vanishing gradients (answer: cell state as a gradient highway). You are also expected to recognize PyTorch tensor shapes: LSTM output = (batch, seq, hidden); h_n, c_n = (num_layers, batch, hidden). Finally, you must trace the path from Word2Vec → ELMo → Transformers to understand why Transformers replaced RNNs (hint: parallelization and long-range dependencies, covered in Lecture 3).

## Prerequisites

- **Lecture 1 concepts**: backpropagation, activation functions (ReLU, sigmoid), overfitting.
- **Matrix multiplication**: dot product, shapes. If A is (m×n) and B is (n×p), then AB is (m×p).
- **Python/PyTorch syntax**: basic familiarity with `nn.Module`, `nn.Linear`, `forward()`. If not, read the PyTorch intro in Lecture 1.

---

## Concept 1: Loss Functions — MSE, Binary Cross-Entropy, Categorical Cross-Entropy

### The problem it solves

Backpropagation needs a scalar "how wrong is my prediction?" signal to compute gradients. Different tasks require different loss functions: MSE for continuous outputs (house price), Binary Cross-Entropy for 2-class probabilities (spam/ham), Categorical Cross-Entropy for multi-class probabilities (cat/dog/bird). Using the wrong loss function breaks training.

### The intuition

**Loss function = report card.**

After every prediction, the loss function scores how far you are from the truth. Gradient descent then adjusts weights to lower that score.

**Why different tasks need different losses:**

1. **Regression (predict a number):**  
   "Predict house price: $300K." If you say $280K, you are $20K off. MSE penalizes large errors more (squared term), so predicting $500K (off by $200K) is 100× worse than $280K (off by $20K).

2. **Binary classification (predict probability):**  
   "Predict spam probability: 0.95." If the true label is spam (1), you should be confident (output close to 1). If you output 0.05 (confident it is NOT spam), you are catastrophically wrong. Binary Cross-Entropy uses *log* to heavily penalize confident mistakes.

3. **Multi-class classification (predict one of K classes):**  
   "Predict: cat, dog, or bird." You output probabilities: [0.7 cat, 0.2 dog, 0.1 bird]. If true label = cat, Categorical Cross-Entropy only cares about the 0.7 (cat probability). The closer to 1.0, the lower the loss.

### Worked example — MSE (Regression)

You predict house prices. Training set:

| Actual (y) | Predicted (ŷ) | Error (y - ŷ) | Squared Error |
|------------|---------------|---------------|---------------|
| 300K       | 280K          | 20K           | 400M          |
| 450K       | 460K          | -10K          | 100M          |
| 200K       | 250K          | -50K          | 2500M         |

MSE = (400M + 100M + 2500M) / 3 ≈ 1000M

The third prediction (off by 50K) dominates the loss (2500M vs. 400M for the first). MSE punishes large errors quadratically.

**Why squared?**

1. Makes all errors positive (negative errors do not cancel positive ones).
2. Penalizes outliers heavily (off by 2× → 4× more loss).
3. Differentiable (needed for gradient descent).

### Worked example — Binary Cross-Entropy (Spam Filter)

You predict spam probability. Test set:

| Email | True Label (y) | Predicted (ŷ) | Loss |
|-------|----------------|---------------|------|
| 1     | Spam (1)       | 0.95          | -log(0.95) ≈ 0.05 (good!) |
| 2     | Spam (1)       | 0.05          | -log(0.05) ≈ 3.0 (disaster!) |
| 3     | Ham (0)        | 0.1           | -log(1-0.1) = -log(0.9) ≈ 0.1 (good!) |

**Why logarithm?**

BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]

For email 2: true label = 1 (spam), but you predicted 0.05 (95% confident it is ham). BCE = -log(0.05) ≈ 3.0. The *log* makes this penalty huge. If you had predicted 0.5 (uncertain), BCE = -log(0.5) ≈ 0.69 (much smaller penalty). The loss *explodes* when you are confidently wrong.

For email 3: true label = 0 (ham), you predicted 0.1 (90% confident it is ham). The term (1-y) log(1-ŷ) = 1 × log(0.9) ≈ -0.1 → BCE ≈ 0.1. Low penalty because you are mostly correct.

### Worked example — Categorical Cross-Entropy (Image Classification)

You classify images into 3 classes: cat, dog, bird. One sample:

True label: cat (one-hot encoded as [1, 0, 0])

Model output (after Softmax): [0.7 cat, 0.2 dog, 0.1 bird]

CCE = -Σ_c y_c log(ŷ_c)  
     = -(1 × log(0.7) + 0 × log(0.2) + 0 × log(0.1))  
     = -log(0.7)  
     ≈ 0.36

**Key insight:** Only the probability of the *true* class matters. The model said "70% cat," but the true label is cat, so you only look at log(0.7). If the model had said "99% cat," loss = -log(0.99) ≈ 0.01 (much better).

### The formula (now with meaning)

**MSE (Mean Squared Error):**

MSE = (1/n) Σ (y_i - ŷ_i)²

Use when: output is a continuous number (regression). Examples: house price, temperature, stock price.

**Binary Cross-Entropy (BCE):**

BCE = -(1/n) Σ [y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i)]

Use when: binary classification (2 classes). Output layer = Sigmoid (range 0–1). Examples: spam/ham, fraud/legit, cancer/healthy.

**Categorical Cross-Entropy (CCE):**

CCE = -(1/n) Σ_i Σ_c y_{i,c} log(ŷ_{i,c})

Use when: multi-class classification (K > 2). Output layer = Softmax (probabilities sum to 1). Examples: MNIST digits, ImageNet classes, sentiment (positive/neutral/negative).

**Output activation pairing (critical):**

| Task               | Loss              | Output Activation |
|--------------------|-------------------|-------------------|
| Regression         | MSE               | None (linear)     |
| Binary             | Binary CE         | Sigmoid           |
| Multi-class        | Categorical CE    | Softmax           |

### Common misunderstandings

- "I can use MSE for classification." Technically yes, but it performs poorly because MSE penalizes all errors equally (predicting cat when truth is dog has the same penalty as predicting bird). Cross-entropy cares about *probability calibration* (confident mistakes are punished exponentially).
- "Binary CE and Categorical CE are the same." True only when K=2. Binary CE is a special case. For K > 2, you must use Categorical CE.
- "Cross-entropy always decreases during training." False. If the model overfits, training CE decreases but validation CE increases.

### Exam angle

MCQs will give you a task and ask which loss function. Trap question: "You are predicting 10 classes. Which loss?" Wrong answer: "Binary Cross-Entropy" (only works for 2 classes). Correct: "Categorical Cross-Entropy." Another trap: "You are predicting continuous values. Which activation for output layer?" Wrong: "Softmax" (gives probabilities, not continuous values). Correct: "None (linear)."

---

## Concept 2: Parameters vs Hyperparameters (Revisited with Training Context)

### The problem it solves

Lecture 1 introduced this briefly. Here we revisit it in the context of training loops and model architecture, because exam questions often ask "Which of these is a hyperparameter?" in a list mixing weights, learning rate, batch size, and biases.

### The intuition

**Parameters** = internal variables the model *learns* during training. You never set these manually.

Examples: weights in `nn.Linear(128, 64)`, biases, convolutional filter values.

**Hyperparameters** = external knobs *you* set before training. The model does not learn these.

Examples: learning rate, number of layers, dropout rate, batch size, number of epochs.

**Analogy:** Training a dog.

- **Parameters** = the dog's learned behaviors (sit, stay, fetch). You do not hard-code these — the dog learns them through training.
- **Hyperparameters** = your training method choices (how many treats per trick? how many training sessions per day?). You decide these before training starts.

### Worked example

```python
import torch.nn as nn
import torch.optim as optim

# Hyperparameters (you choose):
learning_rate = 0.001  # Hyperparameter
num_layers = 3         # Hyperparameter
hidden_size = 128      # Hyperparameter
dropout_rate = 0.5     # Hyperparameter
batch_size = 32        # Hyperparameter
num_epochs = 50        # Hyperparameter

# Model (structure defined by hyperparameters):
model = nn.Sequential(
    nn.Linear(784, hidden_size),  # Parameters: weights (784×128) + biases (128)
    nn.ReLU(),
    nn.Dropout(dropout_rate),
    nn.Linear(hidden_size, 10)    # Parameters: weights (128×10) + biases (10)
)

# Optimizer (uses learning_rate hyperparameter):
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop:
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        loss = loss_fn(model(X_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Updates parameters (weights, biases)
```

**What the optimizer updates:** Weights and biases (parameters). These start as random values (e.g., Xavier initialization) and change every iteration.

**What you tune manually:** learning_rate, dropout_rate, hidden_size, num_epochs (hyperparameters). You run experiments with different values (e.g., dropout_rate = 0.3 vs. 0.5) and pick the best.

### The formula (now with meaning)

**Parameter count for a fully connected layer:**

Parameters = (input_size × output_size) + output_size (bias)

Example: `nn.Linear(784, 128)` has (784 × 128) + 128 = 100,480 parameters.

**Hyperparameter search space:**

You define a grid:

```python
hyperparams = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.3, 0.5],
    'hidden_size': [64, 128, 256]
}
```

This gives 3 × 2 × 3 = 18 experiments. Train 18 models, pick the one with lowest validation loss.

### Common misunderstandings

- "Number of layers is a parameter." False. Layers are part of the architecture (hyperparameter). Parameters are the *weights* inside those layers.
- "Batch size affects model capacity." False. Batch size is a training efficiency knob (larger batch → faster GPU utilization, but noisier gradients). It does not change the model's expressive power.

### Exam angle

MCQs will list [weights, learning_rate, batch_size, biases, dropout_rate, number of epochs] and ask "How many are hyperparameters?" Correct: 4 (learning_rate, batch_size, dropout_rate, num_epochs). Weights and biases are parameters. Trap: counting "number of layers" as a parameter (it is a hyperparameter).

---

## Concept 3: CNNs — Local Connectivity, Convolution, Pooling, Parameter Efficiency

### The problem it solves

A fully connected layer for a 224×224×3 RGB image has (224×224×3) × hidden_size = 150,528 × hidden_size parameters. For hidden_size = 512, that is 77 million parameters *just for the first layer*. CNNs reduce this to a few thousand by using *local connectivity* (each neuron sees only a small patch) and *weight sharing* (same filter slides across the entire image).

### The intuition

**Why fully connected (MLP) fails for images:**

An MLP flattens the image into a 150,528-dimensional vector and connects every input pixel to every hidden neuron. Problems:

1. **Too many parameters:** Millions → slow training, overfitting.
2. **Ignores spatial structure:** Pixel (0,0) and pixel (223,223) are treated as unrelated, even though nearby pixels form edges and textures.
3. **Not translation invariant:** If a cat moves 10 pixels to the right, the MLP sees it as a *completely different* input.

**CNN solution:**

1. **Local connectivity:** Each neuron in the convolutional layer sees only a 3×3 (or 5×5) patch of the input. The rest of the image is invisible to that neuron. This mirrors human vision: you recognize a nose by looking at a small region, not the entire face.

2. **Weight sharing (parameter tying):** The same 3×3 filter slides across the *entire* image. Whether the filter is at position (0,0) or (100,100), it uses the *same* 9 weights. This makes CNNs translation invariant: a cat in the top-left and a cat in the bottom-right are detected by the *same* filter.

3. **Hierarchical feature learning:** Early layers detect edges (3×3 filters find vertical/horizontal lines). Mid layers detect textures (combinations of edges). Deep layers detect object parts (eyes, wheels). Final layer combines parts into classes (cat, car).

**Analogy:** Reading a book.

- **MLP approach:** Memorize every word's position. If the sentence moves one line down, you do not recognize it.
- **CNN approach:** Learn patterns ("the cat sat on") that work *anywhere* on the page. Sliding window (convolution) scans the page; you recognize the pattern regardless of position.

### Worked example — Parameter Count Comparison

**Fully connected (MLP) first layer:**

Input: 224×224×3 = 150,528 pixels.  
Hidden layer: 512 neurons.  
Parameters = 150,528 × 512 + 512 (bias) = 77,070,336 ≈ 77 million.

**Convolutional layer:**

Input: 224×224×3.  
Filter size: 3×3.  
Number of filters: 64.  
Parameters per filter = 3×3×3 (RGB channels) + 1 (bias) = 28.  
Total parameters = 28 × 64 = 1,792.

**Reduction:** 77 million → 1,792 = **43,000× fewer parameters** for the first layer.

### Worked example — Convolution as Dot Product

Input (5×5 grayscale image):

```
1  2  3  0  1
0  1  2  3  0
1  3  2  1  2
2  2  0  1  3
1  0  1  2  1
```

Filter (3×3 edge detector):

```
-1  0  1
-2  0  2
-1  0  1
```

Convolution at position (0,0):

Take the top-left 3×3 patch:

```
1  2  3
0  1  2
1  3  2
```

Dot product (element-wise multiply, then sum):

(-1×1) + (0×2) + (1×3) + (-2×0) + (0×1) + (2×2) + (-1×1) + (0×3) + (1×2)  
= -1 + 0 + 3 + 0 + 0 + 4 - 1 + 0 + 2  
= 7

Output feature map position (0,0) = 7.

Now slide the filter one pixel to the right (stride=1) and repeat. The output is a 3×3 feature map (input 5×5 → output 3×3 because 5 - 3 + 1 = 3).

**What is the filter detecting?**

This is a vertical edge detector. Large positive values appear where the image has a vertical brightness transition (dark on left, bright on right). Horizontal edges would have a different filter.

### Worked example — Pooling (Max Pooling)

After convolution, you have a 4×4 feature map:

```
12  20   8   4
18  36  14   6
 5  11  22  30
 2   7  17  25
```

Apply 2×2 Max Pooling (stride=2):

Divide into 2×2 windows:

**Top-left window:**
```
12  20
18  36
```
Max = 36.

**Top-right window:**
```
8   4
14  6
```
Max = 14.

**Bottom-left window:**
```
5  11
2   7
```
Max = 11.

**Bottom-right window:**
```
22  30
17  25
```
Max = 30.

Output (2×2 after pooling):

```
36  14
11  30
```

**Why pool?**

1. **Dimensionality reduction:** 4×4 → 2×2 (75% smaller). Faster training, fewer parameters in next layer.
2. **Translation invariance:** If the feature (e.g., "cat's nose") moves slightly, max pooling still captures it (you take the *max* over a region, not an exact position).
3. **Prevents overfitting:** Pooling discards spatial precision, forcing the network to focus on *presence* of a feature, not its exact location.

### The formula (now with meaning)

**Convolution output size:**

Output_height = (Input_height - Kernel_height + 2×Padding) / Stride + 1  
Output_width = (Input_width - Kernel_width + 2×Padding) / Stride + 1

Example: Input = 224×224, Kernel = 3×3, Padding = 1, Stride = 1.  
Output = (224 - 3 + 2×1) / 1 + 1 = 224. (Same size due to padding.)

**Pooling output size:**

Output_size = (Input_size - Pool_size) / Stride + 1

Example: Input = 4×4, Pool = 2×2, Stride = 2.  
Output = (4 - 2) / 2 + 1 = 2. (Halved.)

**Parameter count (convolutional layer):**

Parameters = (Kernel_height × Kernel_width × Input_channels × Num_filters) + Num_filters (bias)

Example: 3×3 kernel, 3 input channels (RGB), 64 filters.  
Parameters = (3 × 3 × 3 × 64) + 64 = 1,792.

### Common misunderstandings

- "Convolution reduces the number of channels." False. Convolution changes spatial dimensions (height, width), but the number of output channels = number of filters. If you use 64 filters, output has 64 channels.
- "Pooling has learnable parameters." False. Max pooling and average pooling have *no* parameters (just take max/average). Only convolution has learnable weights.
- "CNNs only work on images." False. CNNs work on any grid-structured data: 1D (time series, audio), 2D (images), 3D (video, medical scans). Text-CNN uses 1D convolution for NLP.

### Exam angle

MCQs will ask: "Why do CNNs have fewer parameters than fully connected layers?" Correct: "Local connectivity and weight sharing." Trap: "CNNs use smaller images" (wrong — image size does not reduce parameters; local receptive fields do). Another question: "What does Max Pooling do?" Correct: "Reduces spatial dimensions by taking the max in each window." Trap: "Increases feature map depth" (wrong — pooling does not change depth; convolution does).

---

## Concept 4: RNNs — Sequential Data, Hidden State, Vanishing Gradients

### The problem it solves

CNNs assume inputs are independent (each image is processed separately). But for sequences (sentences, time series, videos), the *order* matters. "Dog bites man" ≠ "Man bites dog." RNNs maintain a *hidden state* that carries information across time steps, enabling memory.

### The intuition

**Why MLPs and CNNs fail on sequences:**

MLP: Treats each word independently. "The cat sat on the ___" → the model sees "the", "cat", "sat", "on", "mat" as 5 unrelated inputs. It cannot predict "mat" because it does not remember "cat sat on."

CNN: Uses local windows. A 3-word convolution can capture "cat sat on," but cannot capture dependencies 20 words apart ("The company, which was founded in 1995 and has offices in 12 countries, ___" → "announced" requires remembering "company" from 15 words ago).

**RNN solution:**

Maintain a *hidden state* h_t that summarizes everything seen so far. At each time step:

1. Read the current word x_t.
2. Combine x_t with the previous hidden state h_{t-1}.
3. Produce new hidden state h_t = f(x_t, h_{t-1}).
4. Output prediction y_t (optional, for tasks like language modeling).

The hidden state acts as *memory*. h_t encodes "I have seen the words: The, cat, sat, on." When the model sees "the" (x_5), it combines with h_4 (memory of "cat sat on") to predict "mat."

**Analogy:** Watching a movie.

- **MLP:** Each frame is shown in isolation. You do not know what happened in previous frames. You see a man holding a gun — is he the hero or villain? You cannot tell.
- **RNN:** You watch frames in order. Frame 1: man saves a child. Frame 2: man fights gangsters. Frame 3: man holding a gun. Now you know he is the hero because you *remember* frames 1 and 2.

### Worked example — RNN for Sentiment Analysis

Sentence: "The movie was amazing"

Vocabulary: {the, movie, was, amazing, terrible, boring} → token IDs.

Input sequence: [1, 2, 3, 4] (token IDs for "The movie was amazing").

**RNN forward pass:**

Initialize h_0 = [0, 0] (zero vector, 2-dimensional hidden state for simplicity).

**Time step 1 (word = "The"):**

x_1 = embedding("The") = [0.1, -0.2]  
h_1 = tanh(W_h · h_0 + W_x · x_1 + b)  
    = tanh(W_h · [0,0] + W_x · [0.1,-0.2] + b)  
    ≈ [0.3, 0.1] (random example values)

**Time step 2 (word = "movie"):**

x_2 = embedding("movie") = [0.5, 0.3]  
h_2 = tanh(W_h · h_1 + W_x · x_2 + b)  
    = tanh(W_h · [0.3,0.1] + W_x · [0.5,0.3] + b)  
    ≈ [0.6, 0.4]

**Time step 3 (word = "was"):**

h_3 = tanh(W_h · h_2 + W_x · x_3 + b) ≈ [0.7, 0.5]

**Time step 4 (word = "amazing"):**

h_4 = tanh(W_h · h_3 + W_x · x_4 + b) ≈ [0.9, 0.8]

**Final prediction (sentiment):**

y = sigmoid(W_out · h_4 + b_out) ≈ 0.95 → positive sentiment.

The hidden state h_4 encodes the entire sentence. The model learned that "amazing" (in context of "movie was") → positive.

### Worked example — Vanishing Gradient Problem

**Why RNNs struggle with long sequences:**

Backpropagation Through Time (BPTT) computes gradients by unrolling the RNN:

∂L/∂h_0 = (∂L/∂h_T) × (∂h_T/∂h_{T-1}) × ... × (∂h_1/∂h_0)

Each term ∂h_t/∂h_{t-1} involves the weight matrix W_h and the derivative of tanh.

Tanh derivative: tanh'(z) = 1 - tanh²(z) → max value = 1, typically < 0.25 for most z.

For a 50-step sequence:  
Gradient = 0.25^50 ≈ 10^(-30) → vanishing gradient. Weights barely update.

**Consequence:**

The model cannot learn dependencies longer than ~10 time steps. "The company, founded in 1995, ___" → by the time the RNN reaches the blank, it has *forgotten* "company" (gradient too small to propagate back 15 steps).

**LSTM fixes this (next concept).**

### The formula (now with meaning)

**RNN hidden state update:**

h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)

where:
- h_t = hidden state at time t.
- h_{t-1} = hidden state at previous time step.
- x_t = input at time t (word embedding).
- W_h = recurrent weight matrix (hidden-to-hidden).
- W_x = input weight matrix (input-to-hidden).
- tanh = activation function (squashes to [-1, 1]).

**Output (optional, for language modeling):**

y_t = Softmax(W_out · h_t + b_out)

**Loss (sequence classification):**

L = CrossEntropy(y_final, true_label)

For sequence-to-sequence (e.g., translation), you compute loss at *every* time step and sum.

### Common misunderstandings

- "RNNs process all time steps in parallel." False. RNNs are *sequential* — you must compute h_1 before h_2 before h_3. This makes training slow (unlike CNNs, which parallelize over spatial dimensions).
- "Hidden state size = vocabulary size." False. Hidden state size is a hyperparameter (e.g., 128, 256). Vocabulary size determines the embedding layer input, not the hidden state.
- "RNNs always have vanishing gradients." Not always. For short sequences (< 10 steps), vanilla RNNs work fine. The problem is *long* sequences. LSTM fixes this.

### Exam angle

MCQs will ask: "Why do RNNs struggle with sequences longer than 50 time steps?" Correct: "Vanishing gradients prevent learning long-range dependencies." Trap: "RNNs do not have enough parameters" (wrong — parameter count is fine; the issue is gradient flow). Another question: "What is the hidden state?" Correct: "A vector summarizing information from previous time steps." Trap: "The input at time t" (wrong — that is x_t, not h_t).

---

## Concept 5: LSTM — Cell State, Three Gates, Solving Vanishing Gradients

### The problem it solves

Vanilla RNNs forget information after ~10 time steps due to vanishing gradients. LSTMs introduce a *cell state* (C_t) that acts as a "conveyor belt" carrying information across many steps with minimal modification, plus three *gates* (forget, input, output) that control what information to keep, update, or output. This solves vanishing gradients and enables learning dependencies 100+ steps apart.

### The intuition

**Why LSTM works:**

Vanilla RNN: h_t = tanh(W_h · h_{t-1} + W_x · x_t + b).

Every step multiplies by W_h and passes through tanh. After 50 steps, gradients vanish.

**LSTM:** Adds a *cell state* C_t that flows through time with only *additive* updates (not multiplicative). Gradients flow unchanged through the cell state, bypassing the vanishing gradient problem.

**The three gates (learned mechanisms):**

1. **Forget gate (f_t):** Decides what information to *throw away* from the previous cell state.  
   Example: Reading "The company, founded in 1995, recently ___". When you reach "recently," you can *forget* "1995" (not needed anymore).

2. **Input gate (i_t):** Decides what *new* information to store in the cell state.  
   Example: "recently announced" → store "announce" (important for predicting the next word, like "profits").

3. **Output gate (o_t):** Decides what information from the cell state to *output* as the hidden state.  
   Example: After reading "recently announced," the hidden state h_t should encode "announcement context" (relevant for next word), but not the full history (C_t contains everything; h_t is a filtered view).

**Analogy:** Note-taking during a lecture.

- **Cell state (C_t):** Your full notebook. Contains everything you have written so far.
- **Forget gate:** Cross out irrelevant notes ("This slide is optional — skip it").
- **Input gate:** Add new notes ("Definition of gradient descent: ...").
- **Output gate:** Highlight key points for the exam. The hidden state (h_t) is your "exam cheat sheet" (filtered summary), while the cell state (C_t) is the full notebook.

### Worked example — LSTM on Toy Data (1 Time Step)

**Setup:**

Previous hidden state: h_{t-1} = 0.5  
Previous cell state: C_{t-1} = 0.3  
Current input: x_t = 1.0  
All weights = 0.5, all biases = 0.

**Forget gate (f_t):**

f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)  
    = sigmoid(0.5 × [0.5 + 1.0] + 0)  
    = sigmoid(0.75)  
    ≈ 0.679

**Interpretation:** Keep 67.9% of the old cell state C_{t-1}. Forget 32.1%.

**Input gate (i_t):**

i_t = sigmoid(0.5 × [0.5 + 1.0] + 0) ≈ 0.679

**Cell candidate (C̃_t):**

C̃_t = tanh(0.5 × [0.5 + 1.0] + 0)  
     = tanh(0.75)  
     ≈ 0.635

**New cell state (C_t):**

C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  
    = 0.679 × 0.3 + 0.679 × 0.635  
    = 0.204 + 0.431  
    = 0.635

**Interpretation:** We kept 20.4% of the old memory (f_t × C_{t-1}) and added 43.1% new information (i_t × C̃_t). Total = 0.635.

**Output gate (o_t):**

o_t = sigmoid(0.5 × [0.5 + 1.0] + 0) ≈ 0.679

**New hidden state (h_t):**

h_t = o_t ⊙ tanh(C_t)  
    = 0.679 × tanh(0.635)  
    = 0.679 × 0.561  
    ≈ 0.381

**Summary:** One LSTM time step updated the cell state from 0.3 → 0.635 (added new memory) and produced hidden state h_t = 0.381 (filtered output for next layer).

### Worked example — PyTorch LSTM Output Shapes

```python
import torch
import torch.nn as nn

batch_size = 16
seq_length = 10
input_size = 300  # Word embedding dimension
hidden_size = 128
num_layers = 2

lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

x = torch.randn(batch_size, seq_length, input_size)  # (16, 10, 300)

output, (h_n, c_n) = lstm(x)

print(output.shape)  # (16, 10, 128)  → hidden state at EVERY time step
print(h_n.shape)     # (2, 16, 128)    → final hidden state for each layer
print(c_n.shape)     # (2, 16, 128)    → final cell state for each layer
```

**What each tensor means:**

- **output:** Hidden state h_t at *every* time step (t=1, 2, ..., 10). Shape: (batch, seq, hidden). Use for sequence-to-sequence tasks (e.g., translation: output at every step).
  
- **h_n:** Hidden state at the *final* time step (t=10) for *each layer*. Shape: (num_layers, batch, hidden). Use for classification (e.g., sentiment: only the final h_n matters).

- **c_n:** Cell state at the final time step for each layer. Shape: (num_layers, batch, hidden). Used to initialize the next sequence (if processing a long sequence in chunks).

**Critical exam detail:** Students often confuse output and h_n. Remember:  
- `output` = all time steps (use for seq2seq).  
- `h_n` = final time step only (use for classification).

### The formula (now with meaning)

**LSTM equations (4 gates + cell update + hidden update):**

**Forget gate:**  
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  
Range: [0, 1]. 0 = forget all, 1 = keep all.

**Input gate:**  
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  
Range: [0, 1]. Controls how much of the new candidate to write.

**Cell candidate (new information):**  
C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  
Range: [-1, 1]. Candidate values to add to cell state.

**Cell state update:**  
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  
⊙ = element-wise multiply (Hadamard product).

**Output gate:**  
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  
Range: [0, 1]. Filters the cell state to produce hidden state.

**Hidden state update:**  
h_t = o_t ⊙ tanh(C_t)  
Range: [-1, 1] (after tanh).

**Key insight:** The cell state C_t flows through time with only *additive* updates (f_t ⊙ C_{t-1} + i_t ⊙ C̃_t). Gradients flow through addition unchanged (chain rule: ∂(a+b)/∂a = 1). This bypasses vanishing gradients.

### Common misunderstandings

- "LSTM has 4 gates." False. LSTM has 3 gates (forget, input, output) and 1 cell candidate (C̃_t). Total = 4 equations, but only 3 are "gates" (sigmoid-based).
- "Cell state is the output." False. Cell state C_t is internal memory (long-term). Hidden state h_t is the output (short-term, filtered view of C_t).
- "LSTM completely solves vanishing gradients." False. LSTM helps, but for *very* long sequences (1000+ steps), gradients still degrade. Transformers (attention) solve this by connecting every position directly (no sequential dependency).

### Exam angle

MCQs will show LSTM equations and ask "What does the forget gate do?" Correct: "Decides what percentage of the old cell state to keep." Trap: "Decides what to forget from the input x_t" (wrong — forget gate operates on C_{t-1}, not x_t). Another question: "Why does LSTM avoid vanishing gradients?" Correct: "Cell state updates are additive, so gradients flow unchanged." Trap: "LSTM uses ReLU instead of tanh" (wrong — LSTM uses sigmoid for gates and tanh for candidates).

---

## Concept 6: CNN vs RNN — When to Use Which

### The problem it solves

Students often ask: "Should I use CNN or RNN for my task?" The answer depends on the data structure: spatial (images) → CNN; sequential (text, time series) → RNN; or hybrid (video = spatial + temporal) → CNN+RNN.

### The intuition

**CNN strengths:**

- **Spatial data (images, grid-like):** Convolution detects local patterns (edges, textures). Pooling provides translation invariance. Hierarchy: low-level → high-level features.
- **Parallelization:** All spatial positions processed simultaneously (fast on GPU).
- **Example tasks:** Image classification, object detection, semantic segmentation.

**RNN strengths:**

- **Sequential data (text, audio, time series):** Hidden state carries memory across time. Order matters ("dog bites man" ≠ "man bites dog").
- **Variable-length input:** RNNs handle sentences of any length (no fixed window).
- **Example tasks:** Sentiment analysis, machine translation, speech recognition, stock forecasting.

**CNN weaknesses:**

- **Ignores order:** Convolution is a local window. "The cat sat" and "sat cat the" are treated similarly (unless you use 1D convolution with very large kernels).
- **Fixed input size (without tricks):** Standard CNN expects 224×224 images. You must resize or crop.

**RNN weaknesses:**

- **Sequential (slow):** Must process time step 1 before time step 2. Cannot parallelize across time.
- **Vanishing gradients:** Vanilla RNN fails on long sequences. LSTM helps, but still slower than Transformers.

### Worked example — Task Classification

| Task                                      | Best Model | Why?                                                                 |
|-------------------------------------------|------------|----------------------------------------------------------------------|
| Classify 224×224 RGB images (cats/dogs)   | CNN        | Spatial structure. Convolution + pooling detect features.            |
| Predict next word in sentence             | RNN (LSTM) | Sequential. Memory of previous words needed.                         |
| Sentiment analysis ("loved this movie")   | RNN or CNN | RNN: captures word order. CNN: 1D convolution (local phrase patterns).|
| Translate English → French                | RNN (seq2seq) | Encoder RNN (English) → Decoder RNN (French). Seq-to-seq.        |
| Detect pedestrians in video               | CNN + RNN  | CNN extracts features per frame. RNN models temporal motion.         |
| Stock price forecasting                   | RNN (LSTM) | Time series. Past values predict future.                             |
| Segment brain tumor in MRI scan           | CNN (U-Net)| Spatial. Pixel-level prediction (segmentation).                      |

### The formula (now with meaning)

**Parameter count comparison (image input):**

**MLP:**  
Parameters = input_size × hidden_size = 150,528 × 512 ≈ 77M (first layer).

**CNN:**  
Parameters = kernel_size² × input_channels × num_filters = 3² × 3 × 64 = 1,792 (first layer).

**RNN (for sequential text, vocab=10K, embedding=300, hidden=128):**  
Parameters = (embedding × hidden) + (hidden × hidden) = (300 × 128) + (128 × 128) ≈ 54K.

CNN is most parameter-efficient for images. RNN is efficient for sequences.

### Common misunderstandings

- "RNN is always better for NLP." False. Modern NLP uses Transformers (attention), which outperform RNNs. But for small datasets or real-time applications (low latency), RNNs/LSTMs are still used.
- "CNN cannot handle text." False. Text-CNN uses 1D convolution over word embeddings. Works well for short text (e.g., sentence classification). Example: Kim (2014) CNN for sentence classification.

### Exam angle

MCQs will describe a task and ask "Which architecture?" Example: "You have 10,000 time-series samples (hourly temperature for 1 year). Predict next hour's temperature." Correct: RNN (LSTM). Trap: CNN (wrong — time series is sequential, not spatial). Another trap: "Use CNN because it is faster" (true, but RNN is better for temporal dependencies).

---

## Concept 7: Word2Vec Limitations — Static Embeddings and Polysemy

### The problem it solves

Word2Vec gives *one* vector per word, regardless of context. "bank" (river) and "bank" (money) collapse into a single vector, averaging both meanings. The model cannot distinguish polysemy (words with multiple meanings). ELMo (next concept) fixes this with context-dependent embeddings.

### The intuition

**Word2Vec (static):**

During training, Word2Vec sees "bank" in two contexts:

1. "I deposited money at the bank." (finance)
2. "She sat on the river bank." (geography)

Word2Vec averages: bank_vector = 0.5 × finance_vector + 0.5 × geography_vector.

**Problem:** The single vector is a *compromise* that does not fully represent either meaning. If you compute similarity:

```python
model.wv.most_similar("bank")
# [('banks', 0.92), ('banking', 0.89), ('river', 0.65), ...]
```

Both "banking" (finance) and "river" (geography) appear, but neither is strongly represented. The model is confused.

**Real-world consequence:**

Search engine: User searches "bank jobs" (wants finance jobs). But the system also returns "river bank restoration jobs" because "bank" is ambiguous.

**What we need:** A *different* embedding for "bank" in "bank jobs" (finance context) vs. "bank restoration" (river context). This requires *contextual* embeddings.

### Worked example — Word2Vec Cannot Disambiguate

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "deposited", "money", "at", "the", "bank"],
    ["She", "sat", "on", "the", "river", "bank"],
    ["The", "bank", "approved", "my", "loan"],
    ["The", "bank", "of", "the", "river", "is", "muddy"]
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Get the single vector for "bank"
bank_vec = model.wv["bank"]

# This vector is the same for ALL contexts (finance and river)
print(model.wv.most_similar("bank", topn=5))
# Output: [('river', 0.85), ('loan', 0.78), ('money', 0.73), ...]
# Mixed! Both meanings are averaged into one vector.
```

**The issue:** No matter which sentence you process, Word2Vec always uses the *same* bank_vec. You cannot distinguish "bank" in "bank loan" (finance) from "bank" in "river bank" (geography).

### The formula (now with meaning)

**Word2Vec (Skip-gram) objective:**

Maximize: Σ_{(target, context) pairs} log P(context | target)

where P(context | target) = exp(v_context · v_target) / Σ_{all words w} exp(v_w · v_target)

**Key limitation:** v_target is *fixed* for a given word. If "bank" appears in 1000 sentences (500 finance, 500 river), the model learns a single v_bank that averages both contexts.

**What ELMo does differently:**

ELMo computes embeddings *on the fly* using a bidirectional LSTM. For "bank" in "I deposited money at the bank," the LSTM sees {I, deposited, money, at, the} (left context) and {} (no right context beyond "bank"). The embedding is influenced by "deposited" and "money" → finance vector.

For "bank" in "river bank," the LSTM sees {river} (left context) → geography vector.

Same word, *different* vectors.

### Common misunderstandings

- "Word2Vec is obsolete." Not entirely. For small datasets or low-resource languages, Word2Vec is fast and effective. For state-of-the-art NLP (where you need context), use BERT/GPT (Transformers).
- "Polysemy only affects rare words." False. Common words like "bank," "play," "run" have multiple meanings. Polysemy is everywhere.

### Exam angle

MCQs will ask: "Why is Word2Vec limited?" Correct: "It produces static embeddings that cannot distinguish polysemy." Trap: "Word2Vec has too few dimensions" (wrong — you can use 300 or 1000 dimensions; the issue is *static* vectors, not size). Another question: "How does ELMo improve on Word2Vec?" Correct: "ELMo computes context-dependent embeddings using bidirectional LSTM." Trap: "ELMo uses a larger vocabulary" (wrong — vocabulary size is not the issue).

---

## Concept 8: ELMo — Contextual Embeddings with Bidirectional LSTM

### The problem it solves

Word2Vec gives one vector per word (static). ELMo computes a *different* vector for the same word in different contexts by running a bidirectional LSTM over the sentence. "bank" in "river bank" gets a geography-flavored vector, while "bank" in "bank loan" gets a finance-flavored vector.

### The intuition

**ELMo = Embeddings from Language Models (Peters et al., 2018).**

**Key idea:** Train a bidirectional LSTM to predict the next word (forward direction) and the previous word (backward direction). The hidden states at each time step encode context from *both* directions. Use these hidden states as embeddings.

**Why bidirectional?**

Sentence: "I went to the bank."

- **Forward LSTM (left-to-right):** Sees {I, went, to, the} before "bank." Predicts "bank" based on left context. Hidden state encodes "going somewhere."
  
- **Backward LSTM (right-to-left):** Sees nothing after "bank" (end of sentence), but during training it sees the full sentence in reverse. Hidden state encodes "destination."

**Combined:** The ELMo vector for "bank" = concatenation of forward_hidden + backward_hidden. This captures context from *both* sides.

**Contrast with Word2Vec:**

Word2Vec: "bank" → one fixed 300-dim vector (learned during training, never changes).

ELMo: "bank" in "I went to the bank" → 1024-dim vector (computed on the fly, depends on "I went to the").  
       "bank" in "river bank" → different 1024-dim vector (depends on "river").

### Worked example — ELMo in Action

```python
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

sent1 = ["I", "deposited", "money", "at", "the", "bank"]
sent2 = ["She", "sat", "on", "the", "river", "bank"]

emb1 = elmo.embed_sentence(sent1)  # shape: (3, 6, 1024)
emb2 = elmo.embed_sentence(sent2)  # shape: (3, 6, 1024)

# Extract "bank" embedding (position 5 in both sentences, layer 2)
bank_finance = emb1[2][5]  # layer 2, word 5
bank_river = emb2[2][5]    # layer 2, word 5

# Compute cosine similarity
import numpy as np
from numpy.linalg import norm

sim = np.dot(bank_finance, bank_river) / (norm(bank_finance) * norm(bank_river))
print(f"Similarity: {sim:.3f}")  # ~0.54 (different, not orthogonal!)
```

**Interpretation:**

Word2Vec: cos(bank_finance, bank_river) ≈ 1.0 (same vector).

ELMo: cos(bank_finance, bank_river) ≈ 0.54 (different vectors, but still somewhat related because both are "bank").

**Why not 0?** Because both sentences contain "bank" (the word itself is the same). The difference comes from context ("deposited money" vs. "river"). ELMo captures this nuance.

### Worked example — ELMo Tensor Shapes

ELMo output: (num_layers, seq_length, embedding_dim)

For sent1 = ["I", "deposited", "money", "at", "the", "bank"]:

```python
emb1 = elmo.embed_sentence(sent1)
print(emb1.shape)  # (3, 6, 1024)
# 3 layers (bottom layer = char CNN, middle = forward LSTM, top = backward LSTM)
# 6 words in the sentence
# 1024-dim embedding per word per layer
```

**How to use ELMo embeddings:**

1. **Concatenation (default):** Use all 3 layers.  
   word_emb = concat(emb1[0][i], emb1[1][i], emb1[2][i]) → 3072-dim.

2. **Weighted sum:** Learn task-specific weights α_0, α_1, α_2.  
   word_emb = α_0 × emb1[0][i] + α_1 × emb1[1][i] + α_2 × emb1[2][i].

3. **Top layer only:** Use emb1[2][i] (1024-dim). Common for downstream tasks.

### The formula (now with meaning)

**ELMo architecture:**

1. **Character CNN:** Each word is represented as a sequence of characters. A CNN produces a character-aware word embedding (handles out-of-vocabulary words and morphology).

2. **Bidirectional LSTM (2 layers):**
   - Forward LSTM (left-to-right): h_t^fwd = LSTM(h_{t-1}^fwd, x_t).
   - Backward LSTM (right-to-left): h_t^bwd = LSTM(h_{t+1}^bwd, x_t).

3. **Contextualized embedding for word t:**
   ELMo_t = f(h_t^fwd, h_t^bwd, char_CNN(word_t))

   where f is typically a weighted sum of all layers.

**Key difference from Word2Vec:**

Word2Vec: Lookup table. bank → fixed vector.

ELMo: Run LSTM. bank in context → dynamic vector.

### Common misunderstandings

- "ELMo is a Transformer." False. ELMo uses bidirectional LSTM (2018). BERT (also 2018, released later) uses Transformers. Both solve polysemy, but BERT is more powerful.
- "ELMo requires fine-tuning." False. ELMo is *pre-trained* on a large corpus (1 billion words) and used as a feature extractor (frozen). You can fine-tune, but it is optional. BERT *encourages* fine-tuning.
- "ELMo solves all NLP problems." False. ELMo improves over Word2Vec, but Transformers (BERT, GPT) further improve by using attention (better long-range dependencies and parallelization).

### Exam angle

MCQs will ask: "What makes ELMo embeddings contextual?" Correct: "Bidirectional LSTM computes embeddings based on surrounding words." Trap: "ELMo uses a larger vocabulary than Word2Vec" (wrong — vocabulary size is not the key; context is). Another question: "ELMo output shape is (3, 6, 1024). What does 3 represent?" Correct: "Number of layers (char CNN + forward LSTM + backward LSTM)." Trap: "Batch size" (wrong — batch size would be the first dimension in PyTorch, but ELMo processes one sentence at a time by default).

---

## Concept 9: Evolution from Statistical NLP to Transformers

### The problem it solves

You need to understand the *path* from Word2Vec → ELMo → Transformers to grasp why modern LLMs (GPT, BERT, ChatGPT) use attention instead of RNNs. This is not tested in isolation but provides the narrative arc for Lecture 3 (Transformers).

### The intuition

**The timeline:**

1. **Statistical NLP (1990s–2000s):** Hand-crafted features (TF-IDF, n-grams), probabilistic models (HMM, CRF). Good for syntax, bad for semantics.

2. **Deep NLP (2013–2017):** Word2Vec (2013), RNN/LSTM (2015–2017). Learned embeddings from data. RNNs handle sequences. Problem: vanishing gradients + slow (sequential).

3. **Pre-training (2018):** ELMo (Feb 2018), GPT-1 (Jun 2018), BERT (Oct 2018). Key idea: pre-train on massive unlabeled corpora (Wikipedia), then fine-tune on task-specific data. Transfer learning for NLP.

4. **Transformers (2017–present):** "Attention Is All You Need" (Vaswani et al., 2017). Replace RNN with self-attention. Parallelizable (fast). Handles long-range dependencies (no vanishing gradient). Enables GPT-2, GPT-3, ChatGPT, BERT, T5.

5. **Foundation Models (2020–present):** GPT-3 (175B parameters), PaLM (540B), LLaMA, GPT-4. Pre-train once, prompt for many tasks (zero-shot, few-shot). No fine-tuning needed (prompt engineering instead).

**Why this matters for the exam:**

You must recognize the limitations at each stage:

- Word2Vec → polysemy (static embeddings).
- ELMo → sequential (LSTM slow, cannot parallelize).
- Transformers → solve both (context via attention, parallelizable).

### Worked example — The Bottleneck of RNNs

**RNN/LSTM limitations:**

1. **Sequential processing:** Must process word 1 before word 2 before word 3. Cannot parallelize. For a 50-word sentence, this is 50 sequential steps (slow on GPU).

2. **Long-range dependencies:** Even with LSTM, a 100-word sentence struggles to remember word 1 by the time you reach word 100. Attention fixes this by connecting *every* word directly to *every* other word.

3. **Vanishing gradients (even with LSTM):** For 1000+ steps, gradients still degrade. Transformers use residual connections + layer normalization to maintain gradient flow.

**Transformer solution:**

Self-attention: Compute attention scores between *all* pairs of words in parallel. "The company, founded in 1995, recently announced" → "announced" can attend directly to "company" (even though they are 6 words apart) in one step.

No sequential dependency → parallelizable → 10× faster training.

### The formula (now with meaning)

**RNN complexity:**

Time complexity = O(seq_length) (sequential, no parallelization).  
Memory = O(seq_length × hidden_size) (hidden state per time step).

**Transformer complexity:**

Time complexity = O(seq_length²) (all-pairs attention), but parallelizable.  
Memory = O(seq_length² × hidden_size) (attention matrix).

For short sequences (< 512 words), Transformers are faster. For very long sequences (> 4096 words), Transformers require approximations (sparse attention, Longformer).

### Common misunderstandings

- "ELMo is a Transformer." False. ELMo uses LSTM (2018). BERT (also 2018, 6 months later) uses Transformers.
- "Transformers replaced RNNs completely." False in 2024. For real-time streaming (e.g., live transcription), RNNs (especially GRU) are still used because Transformers require the *full* sequence upfront (cannot process one word at a time).
- "Pre-training is the same as transfer learning." Mostly true. Pre-training (train on Wikipedia) → fine-tuning (train on task-specific data) is transfer learning. But modern models (GPT-3) use *zero-shot* prompting (no fine-tuning), which is a different paradigm.

### Exam angle

MCQs will ask: "What is the key advantage of Transformers over RNNs?" Correct: "Self-attention enables parallelization and direct long-range dependencies." Trap: "Transformers have more parameters" (not necessarily — model size is orthogonal to architecture choice). Another question: "Why did ELMo not replace Word2Vec immediately?" Correct: "ELMo is slower (LSTM forward pass per sentence) and requires more compute." Trap: "ELMo is less accurate" (wrong — ELMo is more accurate, but slower).

---

## Putting it together

This lecture builds the foundation for modern deep learning architectures:

1. **Loss functions (MSE, BCE, CCE):** How to measure error for different tasks. Use MSE for regression, cross-entropy for classification.

2. **CNNs:** Convolution (local connectivity + weight sharing) → parameter efficiency. Pooling (spatial invariance). Hierarchy (edges → textures → objects). Best for images.

3. **RNNs:** Hidden state (memory across time steps). Sequential processing. Best for sequences (text, time series). Limitation: vanishing gradients.

4. **LSTMs:** Cell state (gradient highway) + three gates (forget, input, output). Solves vanishing gradients. Enables learning 100+ step dependencies. Output shapes: (batch, seq, hidden); h_n, c_n: (num_layers, batch, hidden).

5. **CNN vs RNN:** CNNs for spatial (images). RNNs for temporal (text, time series). Hybrid for video.

6. **Word2Vec → ELMo:** Word2Vec = static embeddings (polysemy problem). ELMo = contextual embeddings (bidirectional LSTM). "bank" in "river bank" ≠ "bank" in "bank loan."

7. **Evolution to Transformers:** RNN limitations (sequential, slow, vanishing gradients) → Transformers (self-attention, parallelizable, no vanishing gradients). Pre-training (Wikipedia) → fine-tuning (task data) or zero-shot prompting.

The arc: hand-crafted features → learned features (CNNs, Word2Vec) → learned representations (RNNs, ELMo) → learned attention (Transformers). Modern LLMs (GPT-4, Claude) are Transformers at scale.

---

## Check yourself (5 questions)

**Q1.** You are building an image classifier (224×224×3 RGB, 1000 classes). Which loss function and output activation should you use?

A. MSE + Linear (no activation)  
B. Binary Cross-Entropy + Sigmoid  
C. Categorical Cross-Entropy + Softmax  
D. Categorical Cross-Entropy + ReLU

**Answer:** C — Multi-class classification (1000 classes) requires Categorical Cross-Entropy (penalizes wrong class predictions) + Softmax (outputs probabilities that sum to 1, one per class). Option A is wrong (MSE is for regression). Option B is wrong (Binary CE only for 2 classes). Option D is wrong (ReLU is for hidden layers, not output; Softmax is needed for probabilities).

---

**Q2.** A convolutional layer has the following parameters:

- Input: 224×224×3 (RGB image)
- Kernel size: 3×3
- Number of filters: 64
- Stride: 1, Padding: 1

How many learnable parameters does this layer have?

A. 1,728  
B. 1,792  
C. 614,400  
D. 150,528

**Answer:** B — Parameters = (kernel_height × kernel_width × input_channels × num_filters) + num_filters (bias).  
= (3 × 3 × 3 × 64) + 64  
= (9 × 3 × 64) + 64  
= 1,728 + 64  
= **1,792**.

Option A forgot the bias term. Option C is the number of *output* activations (224×224×64 ÷ something), not parameters. Option D is from an MLP first layer (224×224×3 = 150,528 inputs), not a CNN.

---

**Q3.** An LSTM processes a batch of sequences. The batch size is 16, sequence length is 20, input dimension is 300, hidden size is 128, and there are 2 layers. What is the shape of the final hidden state `h_n`?

A. (16, 20, 128)  
B. (2, 16, 128)  
C. (16, 128)  
D. (20, 128)

**Answer:** B — `h_n` is the final hidden state for *each layer*. Shape: (num_layers, batch_size, hidden_size) = (2, 16, 128). Option A is the shape of `output` (hidden state at *every* time step). Option C would be correct if there were only 1 layer (then h_n.squeeze(0) gives (16, 128)). Option D is missing batch dimension.

---

**Q4.** Why do CNNs have far fewer parameters than fully connected (MLP) layers when processing images?

A. CNNs use smaller images.  
B. CNNs use local connectivity (each neuron sees only a small patch) and weight sharing (same filter across the image).  
C. CNNs use ReLU activation, which has no parameters.  
D. CNNs downsample with pooling, reducing the input size.

**Answer:** B — CNNs achieve parameter efficiency through two key mechanisms: (1) *Local connectivity*: each neuron connects to a 3×3 (or 5×5) patch, not the entire image. (2) *Weight sharing*: the same filter weights are reused across all spatial locations. This reduces parameters from millions (MLP) to thousands (CNN). Option A is wrong (image size is the same; the *architecture* changes). Option C is wrong (ReLU has no parameters, but that is true for MLPs too). Option D is wrong (pooling reduces *activations*, not *parameters* of the convolutional layer).

---

**Q5.** A student trains a vanilla RNN (no LSTM) on a sentiment analysis task with 100-word sentences. Training loss decreases, but accuracy plateaus at 65%. The student suspects vanishing gradients. Which change is most likely to help?

A. Increase learning rate from 0.001 to 0.1.  
B. Replace RNN with LSTM.  
C. Add more fully connected layers after the RNN.  
D. Use a smaller vocabulary (reduce embedding size).

**Answer:** B — Vanilla RNNs suffer from vanishing gradients on long sequences (100 words is long). LSTM's cell state acts as a gradient highway, allowing gradients to flow unchanged across many time steps. This enables learning long-range dependencies (e.g., "The movie, despite having great cinematography and acting, was boring" → "boring" is 10 words from "movie," but determines sentiment). Option A is wrong (increasing learning rate to 0.1 will cause divergence, not fix vanishing gradients). Option C is wrong (adding layers after RNN does not fix the gradient flow *through* time steps). Option D is wrong (vocabulary size affects embedding layer, not vanishing gradients; the issue is in the recurrent connections).

---

## Quick reference (for last-day revision)

**Loss Functions:**

| Task               | Loss              | Output Activation | Formula                                    |
|--------------------|-------------------|-------------------|--------------------------------------------|
| Regression         | MSE               | None (linear)     | MSE = (1/n) Σ (y - ŷ)²                     |
| Binary Class       | Binary CE         | Sigmoid           | BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]         |
| Multi-Class        | Categorical CE    | Softmax           | CCE = -Σ_c y_c log(ŷ_c)                    |

**CNNs:**

- **Convolution:** Local connectivity (3×3 patch) + weight sharing (same filter slides across image).  
- **Parameters:** (kernel_h × kernel_w × in_channels × num_filters) + num_filters.  
- **Pooling:** Max or average over window. Reduces spatial dimensions (4×4 → 2×2 via 2×2 pool).  
- **Hierarchy:** Layer 1 (edges) → Layer 2 (textures) → Layer 3 (object parts) → Output (class).  
- **Use for:** Images, spatial data.

**RNNs:**

- **Hidden state:** h_t = tanh(W_h · h_{t-1} + W_x · x_t + b).  
- **Memory:** h_t summarizes all previous time steps.  
- **Problem:** Vanishing gradients for long sequences (> 10 steps). Gradient ≈ 0.25^T → 0.  
- **Use for:** Text, time series, sequences.

**LSTMs:**

- **Cell state (C_t):** Conveyor belt (gradient highway). Additive updates (no multiplication) → gradients flow unchanged.  
- **Three gates:**  
  - **Forget gate (f_t):** What to erase from C_{t-1}. f_t = σ(W_f · [h_{t-1}, x_t] + b_f).  
  - **Input gate (i_t):** What new info to add. i_t = σ(W_i · [h_{t-1}, x_t] + b_i).  
  - **Output gate (o_t):** What to output from C_t. o_t = σ(W_o · [h_{t-1}, x_t] + b_o).  
- **Update:** C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t. h_t = o_t ⊙ tanh(C_t).  
- **PyTorch shapes:**  
  - `output`: (batch, seq, hidden) — all time steps.  
  - `h_n`, `c_n`: (num_layers, batch, hidden) — final step per layer.

**CNN vs RNN:**

| Aspect          | CNN                          | RNN (LSTM)                     |
|-----------------|------------------------------|--------------------------------|
| Best for        | Spatial (images)             | Sequential (text, time series) |
| Parallelization | Yes (across spatial dims)    | No (sequential time steps)     |
| Memory          | None (feedforward)           | Hidden state (memory)          |
| Key operation   | Convolution + pooling        | Recurrent weight update        |

**Word2Vec vs ELMo:**

| Model     | Type             | Polysemy? | Context?  | How?                                |
|-----------|------------------|-----------|-----------|-------------------------------------|
| Word2Vec  | Static           | No        | No        | Lookup table (one vector per word)  |
| ELMo      | Contextual       | Yes       | Yes       | Bidirectional LSTM (different vector per context) |

**Evolution:**

1. Statistical NLP (HMM, CRF) → hand-crafted features.  
2. Word2Vec (2013) → learned embeddings (static).  
3. RNN/LSTM (2015) → sequential modeling (slow, vanishing gradients).  
4. ELMo (2018) → contextual embeddings (LSTM, polysemy solved).  
5. Transformers (2017) → self-attention (parallel, no vanishing gradients).  
6. Foundation Models (GPT-3, 2020) → pre-train once, prompt for many tasks.

**Parameters vs Hyperparameters:**

- **Parameters:** Learned (weights, biases). Optimizer updates these.  
- **Hyperparameters:** Set by you (learning rate, layers, dropout rate, batch size, epochs).

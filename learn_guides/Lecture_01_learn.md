# Lecture 1 — Foundations of Machine Learning and Deep Learning
**Estimated read time:** 30–40 min · **Difficulty:** Foundation

## What this lecture is about

This lecture introduces you to the three core layers of modern artificial intelligence: AI (the goal), Machine Learning (the method), and Deep Learning (the technique). More importantly, it teaches you the practical foundations you need to evaluate models: how to measure success with confusion matrices, precision, recall, and F1 scores; how to avoid overfitting through train-test splits and regularization; and how to represent data efficiently, from sparse one-hot encodings to dense word embeddings. 

This matters for the exam because the proctored questions will test your ability to choose the right metric for a scenario (would you prioritize precision or recall for a cancer detector?), identify overfitting from loss curves, and understand why modern NLP uses embeddings instead of one-hot vectors. You are not just memorizing formulas — you are learning to think like a machine learning engineer debugging a real system.

## Prerequisites

- **Python basics**: loops, functions, numpy arrays.
- **High school algebra**: dot products, sums, exponents (we will build from here).
- **Vague sense of neural networks**: you know "layers" exist and "training" happens. If not: a neural network is a function with adjustable weights that learns patterns from examples.

---

## Concept 1: AI, Machine Learning, Deep Learning — The Hierarchy

### The problem it solves

People use "AI," "Machine Learning," and "Deep Learning" interchangeably in casual conversation. That creates confusion when reading technical papers or job descriptions. Understanding the hierarchy clarifies what each term actually means and where modern LLMs fit in.

### The intuition

Think of these as nested Russian dolls:

- **Artificial Intelligence (AI)** is the outermost goal: build systems that act rationally to achieve outcomes. This includes everything from rule-based chess engines (1950s) to ChatGPT (2020s).
- **Machine Learning (ML)** is a subset of AI: systems that *learn* from data rather than following hard-coded rules. Instead of writing `if spam_word then label=spam`, you show the system 10,000 labeled emails and it learns the pattern.
- **Deep Learning (DL)** is a subset of ML: multi-layer neural networks that learn features automatically from raw data (pixels, text) without manual feature engineering.

The key shift from old AI to ML: instead of a human expert writing rules, the *data* teaches the system. The shift from ML to DL: instead of a human engineer designing features (like "count words ending in -ing"), the *network layers* discover features automatically.

### Worked example

Imagine building a spam filter:

**Old AI (rule-based):**  
```
if "free money" in email: return SPAM
if sender in blacklist: return SPAM
```
This breaks when spammers write "fr33 m0ney."

**Machine Learning (supervised):**  
Train a logistic regression model on 10,000 labeled emails. Features = word counts. Model learns: "free" weight = +2.3, "meeting" weight = -1.1. No explicit rules — the model infers patterns from data.

**Deep Learning (neural network):**  
Feed raw email text into a 3-layer neural network. First layer learns character patterns ("fr33" → "free"), second layer learns word meanings, third layer predicts spam/ham. No manual feature design — the network learns end-to-end.

### The formula (now with meaning)

Tom Mitchell (1997) defined Machine Learning formally:

> A program learns from experience E with respect to task T and performance measure P, if its performance on T, as measured by P, improves with experience E.

For our spam filter:
- **Task (T)**: classify email as spam or not-spam.
- **Experience (E)**: 10,000 labeled training emails.
- **Performance (P)**: accuracy on held-out test emails.

After training on E, the model's P on T goes from 50% (random guessing) to 98% (learned).

### Common misunderstandings

- "Deep Learning is always better than ML." False. For tabular data with 50 features and 1,000 samples, a gradient boosted tree often beats a neural network. DL excels when you have *massive* data (millions of samples) and *high-dimensional* inputs (images, text).
- "AI requires a neural network." False. AlphaGo uses neural networks, but a rules-based chess engine (1980s) is also AI — it acts rationally to win.
- "All modern AI is deep learning." False. Production systems often use ensembles: DL for feature extraction, then XGBoost for final prediction.

### Exam angle

MCQs will give you a scenario and ask which category it belongs to. The trap: confusing "symbolic AI" (rules, logic) with "ML" (data-driven). Example: "An expert system that diagnoses diseases using if-then rules encoded by doctors" → this is classic AI, *not* ML, because it does not learn from data.

---

## Concept 2: Supervised vs Unsupervised Learning

### The problem it solves

Most ML tasks fall into two buckets: you either have labels (supervised) or you don't (unsupervised). Knowing which bucket you are in determines your algorithm choice and how you measure success.

### The intuition

**Supervised Learning:**  
You have a dataset where every example comes with a "right answer" label. Like flashcards: "cat.jpg → cat", "dog.jpg → dog". You train a model to learn the mapping from input (image) to label (class). Then test on new images. 

Common tasks:
- **Classification**: predict a category (spam/ham, cat/dog, cancer/benign).
- **Regression**: predict a number (house price, tomorrow's temperature).

**Unsupervised Learning:**  
You have data but *no* labels. Like dumping 1,000 photos on a table and asking "find patterns." The model discovers hidden structure.

Common tasks:
- **Clustering**: group similar items (K-Means groups customers by behavior, though you never told it what "behavior types" are).
- **Dimensionality Reduction**: compress 10,000 features into 2 for visualization (PCA).
- **Anomaly Detection**: flag outliers (fraud detection).

The key difference: supervised learning needs a "teacher" (labels), unsupervised learning is self-directed exploration.

### Worked example

**Supervised — Breast Cancer Prediction:**

You have 569 tumor samples. Each has 30 measurements (radius, texture, smoothness, ...) and a label: malignant (0) or benign (1).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = data.data  # 569 samples × 30 features
y = data.target  # 569 labels (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)  # Learn from labeled examples

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")  # ~96%
```

Why supervised? You have labels (malignant/benign). You are learning a function f: measurements → label.

**Unsupervised — Recommender System:**

Amazon's "Customers who bought X also bought Y" does NOT use labels like "user123 should see product456." Instead, it clusters users by purchase history: "These 1,000 users all bought detective novels, so show them the new Agatha Christie." No explicit labels — the algorithm discovers the "detective-novel cluster" by finding patterns in purchase co-occurrences.

### The formula (now with meaning)

**Supervised:**  
Given training set {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}, learn f such that f(x) ≈ y for new x.

**Unsupervised:**  
Given data {x₁, x₂, ..., xₙ} (no y!), discover structure. For K-Means clustering, minimize within-cluster variance:  
J = Σₖ Σ_{x in cluster k} ||x − μₖ||²  
where μₖ is the centroid of cluster k.

### Common misunderstandings

- "Clustering is supervised because it groups data." Wrong. Clustering *infers* groups from data. If you already knew the groups and had labels, you would do classification (supervised).
- "Unsupervised learning is useless because there are no labels." Wrong. PCA reduces 10,000 gene expressions to 50 components for cancer research. Autoencoders compress images. Both are unsupervised and extremely useful.

### Exam angle

MCQs will describe a task and ask supervised or unsupervised. The trap: "A model groups customers into 5 segments" sounds supervised because there are 5 groups, but if the model *chose* 5 on its own (not told by a human label), it is unsupervised clustering. The key question: "Did the training data include the right answer?"

---

## Concept 3: Train-Test Split — Why Holdout Data Matters

### The problem it solves

A model that memorizes training data can have 100% training accuracy but 50% test accuracy. Train-test split detects this by simulating "unseen data" *before* you deploy to production.

### The intuition

Imagine studying for an exam by memorizing the practice test answers. If the real exam has the exact same questions, you ace it. But if the real exam has *different* questions on the same topics, you fail — you never learned the underlying concepts.

Machine learning models do the same thing. If you train and evaluate on the *same* data, the model can cheat by memorizing specific examples rather than learning general patterns. The solution: split your data into two sets:

- **Training set (80%)**: model learns from this.
- **Test set (20%)**: model never sees this during training. Used only for final evaluation.

If training accuracy is 99% but test accuracy is 70%, your model memorized the training set (overfitting).

### Worked example

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # first 5 are class 0, last 5 are class 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}")  # 8 samples
print(f"Test size: {len(X_test)}")    # 2 samples
```

Why `random_state=42`? Makes the split reproducible. Without it, every run shuffles differently, making comparisons impossible.

**What happens if you skip the split?**

```python
model.fit(X, y)           # Train on ALL data
accuracy = model.score(X, y)  # Evaluate on same data
# → Accuracy = 100%  (the model just memorized)
```

Deploy this to production and it fails on real users, because real users are *not* in the training set.

### The formula (now with meaning)

Standard split ratio:  
Training set = 80% of data  
Test set = 20% of data

For small datasets (< 1,000 samples), use **cross-validation** instead: split into 5 folds, train on 4, test on 1, repeat 5 times, average results. This gives a more stable estimate.

**Cross-validation error:**  
CV_error = (1/k) Σᵢ₌₁ᵏ errorᵢ  
where k = number of folds (typically 5 or 10).

### Common misunderstandings

- "I split once, so I am safe from overfitting." Partially true. You also need to avoid *peeking* at the test set during development. If you tune hyperparameters by checking test accuracy, you are leaking information. Proper workflow: Train set (fit model) → Validation set (tune hyperparameters) → Test set (final evaluation, touch ONCE).
- "80-20 is a law." No. For massive datasets (millions), 98-2 works fine. For tiny datasets (100 samples), use cross-validation instead of a single split.

### Exam angle

MCQs will show a training curve with train accuracy = 98%, test accuracy = 60%, and ask "What is wrong?" Answer: overfitting. The trap option: "underfitting" (wrong — underfitting has low train AND test accuracy). Another trap: "The test set is too small" (wrong — the gap itself is the problem, not the size).

---

## Concept 4: Confusion Matrix — The Foundation of All Classification Metrics

### The problem it solves

"Accuracy = 95%" sounds great, but what if 95% of your data is class A and your model *always* predicts A? It gets 95% accuracy by doing nothing useful. The confusion matrix breaks down *where* the model is wrong, exposing failures that raw accuracy hides.

### The intuition

A confusion matrix is a 2×2 table (for binary classification) that compares predictions to ground truth:

|                     | **Predicted Positive** | **Predicted Negative** |
|---------------------|------------------------|------------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

Read it like this:
- **True Positive (TP)**: model said "cancer" and it *was* cancer. Good!
- **False Positive (FP)**: model said "cancer" but it was benign. False alarm.
- **False Negative (FN)**: model said "benign" but it was cancer. Missed case — dangerous!
- **True Negative (TN)**: model said "benign" and it was benign. Good!

The diagonal (TP, TN) is correct predictions. Off-diagonal (FP, FN) is errors.

### Worked example — Spam Filter

You test your spam filter on 100 emails:

- 10 are actual spam.
- 90 are actual ham (not spam).

Your model predicts:
- 8 spam (correct) + 2 spam (wrong → FP: flagged ham as spam).
- 2 ham (correct → should have been spam, but missed → FN).
- 88 ham (correct).

Confusion matrix:

|                     | Predicted Spam | Predicted Ham |
|---------------------|----------------|---------------|
| **Actual Spam**     | 8 (TP)         | 2 (FN)        |
| **Actual Ham**      | 2 (FP)         | 88 (TN)       |

Now calculate metrics (next concept), but first, notice:
- The model missed 2 spam emails (FN = 2). If this is a banking fraud detector, those 2 missed cases cost millions.
- The model flagged 2 legitimate emails as spam (FP = 2). Users get annoyed.

You cannot see this from accuracy alone. Accuracy = (TP + TN) / Total = (8 + 88) / 100 = 96%. Sounds great, but you missed 20% of spam (2 out of 10).

### The formula (now with meaning)

Each cell has a name:

- **TP** (True Positive): correctly predicted positive class.
- **TN** (True Negative): correctly predicted negative class.
- **FP** (False Positive): predicted positive, but was actually negative. Type I error.
- **FN** (False Negative): predicted negative, but was actually positive. Type II error.

Total samples = TP + TN + FP + FN.

### Common misunderstandings

- "High accuracy means good model." False. If 99% of emails are ham and your model *always* predicts ham, accuracy = 99% but it catches zero spam. The confusion matrix exposes this: TP = 0.
- "I should minimize both FP and FN equally." Not always. Context matters. For cancer screening, FN (missed cancer) is worse than FP (false alarm → do a follow-up test). For spam, FP (legitimate email in spam folder) is worse than FN (one spam in inbox).

### Exam angle

MCQs will give you a confusion matrix and ask "How many False Negatives?" or "What is the model's worst failure?" The trap: confusing FP and FN. Mnemonic: FN is *failing to catch* the positive class (missed cancer, missed spam). FP is *falsely accusing* a negative case (healthy person flagged as sick).

---

## Concept 5: Precision, Recall, F1 — When to Prioritize What

### The problem it solves

Accuracy treats all errors equally, but in the real world, missing a cancer case (FN) is worse than a false alarm (FP). Precision and recall let you quantify this trade-off and choose the metric that matches your business cost.

### The intuition

Think of a metal detector at an airport:

**High Recall (catch all threats):**  
Beep at *everything* suspicious. You will catch all weapons (no FN), but also flag many innocent items (high FP). Every pocket knife, belt buckle, and coin triggers a search. Long lines, annoyed passengers.

**High Precision (only beep when confident):**  
Beep only when 99% sure it is a weapon. Fewer false alarms (low FP), but you might miss a hidden blade (high FN).

You cannot maximize both. The trade-off is tuned by adjusting the model's threshold.

Now map this to formulas:

- **Precision**: "Of all items I flagged, how many were actually threats?" = TP / (TP + FP)
- **Recall**: "Of all actual threats, how many did I catch?" = TP / (TP + FN)

### Worked example — Cancer Screening

You screen 1,000 patients. 50 have cancer (actual positive), 950 are healthy (actual negative).

Your model flags 60 patients as "high risk":
- 45 of those 60 actually have cancer (TP = 45).
- 15 of those 60 are false alarms, healthy (FP = 15).
- 5 cancer patients were missed, flagged as low risk (FN = 5).
- 940 healthy patients correctly identified as low risk (TN = 940).

Confusion matrix:

|                     | Predicted Cancer | Predicted Healthy |
|---------------------|------------------|-------------------|
| **Actual Cancer**   | 45 (TP)          | 5 (FN)            |
| **Actual Healthy**  | 15 (FP)          | 940 (TN)          |

**Precision:**  
Of the 60 people I flagged, how many actually have cancer?  
Precision = TP / (TP + FP) = 45 / (45 + 15) = 45 / 60 = 0.75 = 75%

Meaning: 75% of flagged patients are true positives. 25% are false alarms who will undergo unnecessary biopsies.

**Recall:**  
Of the 50 actual cancer patients, how many did I catch?  
Recall = TP / (TP + FN) = 45 / (45 + 5) = 45 / 50 = 0.90 = 90%

Meaning: I caught 90% of cancers. But I missed 10% (5 patients go home thinking they are healthy).

**Which is worse?** In cancer screening, missing a case (FN) can be fatal. You prioritize *recall* (catch all cancers) even if it means more false alarms (lower precision). In contrast, for spam filtering, a false alarm (legitimate email in spam folder) loses customer trust, so you prioritize *precision*.

**F1 Score:**  
When you care about *both* precision and recall equally, use F1, the harmonic mean:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

For our cancer model:  
F1 = 2 × (0.75 × 0.90) / (0.75 + 0.90) = 2 × 0.675 / 1.65 = 1.35 / 1.65 ≈ 0.818 = 81.8%

Why harmonic mean instead of arithmetic mean? Because harmonic mean punishes *imbalance*. If precision = 0.1 and recall = 1.0, arithmetic mean = 0.55 (looks okay), but harmonic mean = 0.18 (correctly shows the model is broken).

### The formula (now with meaning)

**Precision** = TP / (TP + FP)  
Denominator = "all I predicted as positive." Answers: "How many of my positive predictions were correct?"

**Recall** = TP / (TP + FN)  
Denominator = "all actual positives." Answers: "How many actual positives did I catch?"

**F1** = 2 × (P × R) / (P + R)  
Balances both. Use when you cannot prioritize one over the other.

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)  
Only use when classes are balanced (50-50 split). Useless for imbalanced datasets like fraud (0.1% fraud, 99.9% legit).

### Common misunderstandings

- "High precision means good model." Not if recall is 10%. A model that flags *one* email as spam and gets it right has precision = 100%, but recall = 1% (missed 99% of spam).
- "F1 is always the best metric." False. If false negatives cost 100× more than false positives (cancer screening), optimize recall directly, not F1.
- "Accuracy and F1 are the same thing." False. For imbalanced data (1% fraud), a model that always predicts "not fraud" has accuracy = 99% but F1 = 0% (because precision and recall are 0).

### Exam angle

MCQs will give you a scenario and ask which metric to optimize. Common trap: "A self-driving car detecting pedestrians. Which metric is critical?" Wrong answer: precision (low false alarms). Correct answer: recall (catch all pedestrians, even if you brake for a cardboard box occasionally). Missing a pedestrian (FN) is fatal. False alarm (FP) is just an abrupt stop.

Another trap: "Model A: precision=0.9, recall=0.5. Model B: precision=0.6, recall=0.9. Which is better?" Answer: depends on the cost structure. No universal answer without context. If the question says "for fraud detection," choose high recall (catch all fraud). If it says "for spam filtering," choose high precision (don't flag legitimate emails).

---

## Concept 6: Feature Scaling — Why Gradient Descent Cares About Units

### The problem it solves

If feature 1 is "house size in square feet" (range: 500–5000) and feature 2 is "number of bedrooms" (range: 1–5), gradient descent will oscillate wildly, taking thousands of iterations to converge. Scaling both features to the same range (e.g., mean=0, std=1) makes gradient descent converge 10× faster.

### The intuition

Imagine you are hiking down a mountain (minimizing loss) in thick fog (you can only see the gradient, not the destination). The slope in the X direction (house size) is 1000× steeper than the slope in the Y direction (bedrooms) because the units are different.

Without scaling, you take huge steps in the X direction and tiny steps in the Y direction, zigzagging back and forth like a pinball. This is slow.

With scaling, both directions have similar slopes, so you can walk straight toward the minimum. Gradient descent converges in 10 steps instead of 1,000.

**Which algorithms care?**
- **Scale-sensitive**: Logistic regression, SVM, neural networks, K-Means (anything using distance or gradient descent).
- **Scale-insensitive**: Tree-based models (decision trees, random forests, XGBoost) because they split on feature *order*, not absolute values.

### Worked example

```python
from sklearn.preprocessing import StandardScaler

X_train = [[1000, 2], [1500, 3], [2000, 4]]  # [house_size, bedrooms]
X_test = [[1200, 2]]

# Without scaling: house_size dominates (range 1000-2000), bedrooms ignored (range 2-4)
# Gradient descent will take 1000+ iterations.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NEVER fit on test set!

print(X_train_scaled)
# [[-1.22, -1.22],
#  [ 0.00,  0.00],
#  [ 1.22,  1.22]]
# Now both features have mean ≈ 0, std ≈ 1. Gradient descent converges fast.
```

**Critical mistake:** `scaler.fit(X_test)` → NEVER do this. The scaler must learn mean and std from *training* data only, then apply the same transformation to test data. Otherwise, you leak information from the test set into training.

### The formula (now with meaning)

**StandardScaler (Z-score normalization):**

z = (x − μ) / σ

where:
- x = original feature value.
- μ = mean of the training set.
- σ = standard deviation of the training set.

After transformation, the feature has mean = 0 and std = 1.

**MinMaxScaler (range normalization):**

x_scaled = (x − x_min) / (x_max − x_min)

Scales everything to [0, 1]. Useful when you know the data has a fixed range (e.g., pixel values 0–255).

### Common misunderstandings

- "I should scale the test set independently." Wrong. You must use the *training* set's mean and std on the test set. Otherwise, test preprocessing differs from training, breaking the model.
- "Scaling changes the model." False. Scaling changes the *units* but not the relationships. A linear model on scaled data is equivalent to the same model on raw data (just with different coefficient magnitudes).
- "I should scale categorical variables." No. One-hot encoded categories (0 or 1) should not be scaled. Scaling makes sense for continuous features only.

### Exam angle

MCQs will ask: "Which preprocessing step is essential before training a neural network?" Correct: StandardScaler. Trap: "PCA" (wrong — PCA is dimensionality reduction, not scaling, though PCA *internally* requires scaling). Another trap: "Use MinMaxScaler for logistic regression" (not wrong, but StandardScaler is more common because it handles outliers better).

---

## Concept 7: Sparse vs Dense Representations — Why One-Hot Fails

### The problem it solves

If you have a vocabulary of 50,000 words and represent each word as a one-hot vector (49,999 zeros and one 1), your model has to store and process massive matrices (50,000 × 50,000), and the dot product of any two words is *always* zero — meaning the model can never learn that "cat" and "kitten" are similar. Dense embeddings fix both problems: they compress 50,000 dimensions to 300, and similar words have similar vectors.

### The intuition

**Sparse (one-hot):**

Vocabulary: {cat, dog, king, queen, apple}

"cat" = [1, 0, 0, 0, 0]  
"dog" = [0, 1, 0, 0, 0]  
"king" = [0, 0, 1, 0, 0]

Dot product of "cat" and "dog" = 1×0 + 0×1 + 0×0 + 0×0 + 0×0 = 0. The model thinks cat and dog are *completely unrelated* because the vectors are orthogonal by construction.

**Dense (embedding):**

"cat"   = [0.8, 0.3, -0.1]  
"dog"   = [0.7, 0.4, -0.2]  
"king"  = [0.1, -0.5, 0.9]

Dot product of "cat" and "dog" = 0.8×0.7 + 0.3×0.4 + (-0.1)×(-0.2) = 0.56 + 0.12 + 0.02 = 0.70. High similarity! The model learns that cat and dog are related (both animals).

### Worked example

**One-hot encoding in sklearn:**

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

vocab = [["cat"], ["dog"], ["king"], ["queen"]]
encoder = OneHotEncoder(sparse_output=False)
one_hot = encoder.fit_transform(vocab)

print(one_hot)
# [[1, 0, 0, 0],
#  [0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]

# Cosine similarity of "cat" and "dog":
from numpy.linalg import norm
cos_sim = np.dot(one_hot[0], one_hot[1]) / (norm(one_hot[0]) * norm(one_hot[1]))
print(cos_sim)  # 0.0 — completely orthogonal!
```

**Problem:** The model cannot learn that "cat" and "dog" are both animals because their vectors are orthogonal.

**Dense embedding in PyTorch:**

```python
import torch
import torch.nn as nn

vocab_size = 50000
embed_dim = 300

embedding = nn.Embedding(vocab_size, embed_dim)

# Token IDs: 42 = "cat", 1337 = "dog"
token_ids = torch.tensor([42, 1337])
vectors = embedding(token_ids)  # shape: (2, 300)

# After training, similar words have similar vectors:
cos_sim = torch.cosine_similarity(vectors[0], vectors[1], dim=0)
print(cos_sim)  # ~0.85 after training (learned similarity)
```

**Result:** The embedding layer *learns* that "cat" and "dog" are similar by adjusting weights during training. The geometry of the vector space encodes meaning.

### The formula (now with meaning)

**One-hot:** Vocabulary size = V. Each word is a V-dimensional vector with one 1 and V-1 zeros.

Memory: V² for weight matrix (first layer of network).

For V = 50,000, this is 2.5 billion parameters just for the first layer → 10GB of RAM.

**Dense embedding:** Each word is a d-dimensional vector (typically d = 300).

Memory: V × d parameters.

For V = 50,000, d = 300, this is 15 million parameters → 60MB of RAM.

Compression ratio: (V²) / (V×d) = V / d = 50,000 / 300 ≈ 167× smaller.

**Why it works:**  
The distributional hypothesis (Firth, 1957): "You shall know a word by the company it keeps." Words that appear in similar contexts ("cat" and "kitten" both appear near "pet," "fur," "meow") get similar vectors. The embedding layer learns this by predicting context words (Word2Vec) or masked words (BERT).

### Common misunderstandings

- "One-hot is bad." Not always. For 3 categories (small, medium, large), one-hot is fine: [1,0,0], [0,1,0], [0,0,1]. The problem is *high-cardinality* categories (50,000 words, 1 million product IDs).
- "Embeddings are always better." False. For structured tabular data (age, income, zip code), embeddings do not help. Use embeddings for *discrete, high-cardinality* data (words, user IDs, product IDs).

### Exam angle

MCQs will ask: "Why does Word2Vec outperform one-hot encoding?" Correct: "One-hot vectors are orthogonal, so the model cannot learn similarity." Trap: "Word2Vec is deeper" (wrong — depth is not the reason; *density* is). Another trap: "One-hot vectors are too large" (partially true, but the *key* issue is orthogonality, not size).

---

## Concept 8: Distributional Hypothesis — Words by Context

### The problem it solves

How do you teach a machine that "cat" and "kitten" are similar, or that "bank" has two meanings (river vs. money)? You cannot hand-code a dictionary of 50,000 words. The distributional hypothesis lets the machine *infer* meaning from context: words that appear near each other in text tend to have related meanings.

### The intuition

Imagine you are learning a new language and see these sentences:

1. "The **blorp** sat on the mat."
2. "The **blorp** drank milk."
3. "I petted the **blorp**."

You have never seen "blorp" before, but from context (sits, drinks milk, gets petted), you infer: blorp is probably a small animal, like a cat.

This is the distributional hypothesis: **meaning = context**. You do not need a definition of "blorp" — you infer it from the words that surround it.

Machine learning models do the same. Word2Vec trains on millions of sentences, counting how often each word appears near other words. Words with similar context distributions (like "cat" and "kitten," which both appear near "pet," "fur," "meow") get similar vectors.

### Worked example

Consider the word "banking" in three contexts:

1. "…government debt problems turning into **banking** crises…"  
   Context words: government, debt, problems, crises → *finance*.

2. "…Europe needs unified **banking** regulation…"  
   Context words: Europe, unified, regulation → *finance*.

3. "…India has just given its **banking** system a shot in the arm…"  
   Context words: India, system, shot, arm → *finance*.

A Word2Vec model sees "banking" appears near {government, debt, regulation, system} thousands of times across a large corpus. It never appears near {river, shore, water}. So the model learns: "banking" (finance) is related to {regulation, debt, crisis}.

Now compare "bank" (river):

1. "…the river **bank** was muddy…"  
   Context: river, muddy, shore.

If the corpus contains *both* senses, Word2Vec averages them into one vector (this is a limitation — Word2Vec cannot distinguish polysemy). Later models like ELMo and BERT solve this by computing context-dependent embeddings.

### The formula (now with meaning)

**Context window (k = 2):**

Given sentence: "The cat sat on the mat"

For target word "sat", context = {cat, on} (2 words before + 2 words after).

Word2Vec learns by predicting context from target (Skip-gram) or target from context (CBOW).

**Skip-gram objective (simplified):**

Maximize: Σ_{context words c} log P(c | target)

where P(c | target) = exp(v_c · v_target) / Σ_{all words w} exp(v_w · v_target)

Intuition: Make the dot product v_c · v_target *high* for words that actually appear near "target" in the corpus, and *low* for random words.

### Common misunderstandings

- "The distributional hypothesis only works for synonyms." False. It also captures analogies (king - man + woman ≈ queen) and semantic relationships (Paris - France + Germany ≈ Berlin).
- "Context window size does not matter." False. Small window (k=2) captures syntax and local associations ("quick brown"). Large window (k=10) captures topical associations ("president → election").

### Exam angle

MCQs will quote Firth (1957): "You shall know a word by the company it keeps," then ask what this enables. Correct: "Learning word similarity from co-occurrence." Trap: "Defining words using dictionaries" (wrong — that is symbolic AI, not distributional). Another trap: "Parsing grammar" (wrong — distributional hypothesis is about meaning, not syntax).

---

## Concept 9: Perceptron, MLP, and Activation Functions

### The problem it solves

A single perceptron (linear model) can only learn linearly separable patterns (like AND, OR). It cannot learn XOR (non-linear). Multi-layer perceptrons (MLPs) with *activation functions* can learn arbitrarily complex non-linear boundaries, enabling deep learning.

### The intuition

**Perceptron (1958):**

A perceptron is a single neuron that computes a weighted sum of inputs, then applies a step function:

z = w₁x₁ + w₂x₂ + b  
ŷ = 1 if z > 0 else 0

This draws a straight line (decision boundary) in 2D, a plane in 3D. It can separate classes only if they are *linearly separable*.

**XOR problem:**

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

Try drawing a single straight line that separates (0,0) and (1,1) from (0,1) and (1,0). Impossible. XOR is not linearly separable.

**MLP (Multi-Layer Perceptron):**

Add a *hidden layer* between input and output. Each neuron in the hidden layer learns a different linear boundary. The output layer combines them. With 2 hidden neurons, an MLP can solve XOR:

- Hidden neuron 1: learns "x₁ OR x₂" (curved boundary).
- Hidden neuron 2: learns "NOT (x₁ AND x₂)" (another curved boundary).
- Output neuron: combines them to learn XOR.

**Activation functions:**

Without a non-linear activation function, stacking layers is pointless. Why? Because a composition of linear functions is still linear:

f(x) = W₂(W₁x) = (W₂W₁)x = W_new x  (still linear)

Adding a non-linearity (sigmoid, ReLU, tanh) between layers breaks this. Now you can approximate *any* function (universal approximation theorem).

### Worked example

**Perceptron for AND gate:**

| x₁ | x₂ | AND |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 0   |
| 1  | 0  | 0   |
| 1  | 1  | 1   |

Perceptron weights: w₁ = 1, w₂ = 1, b = -1.5

z = 1×x₁ + 1×x₂ - 1.5

For (0,0): z = -1.5 → ŷ = 0 ✓  
For (0,1): z = -0.5 → ŷ = 0 ✓  
For (1,0): z = -0.5 → ŷ = 0 ✓  
For (1,1): z = 0.5 → ŷ = 1 ✓

**MLP for XOR:**

You need at least 1 hidden layer with 2 neurons. PyTorch code:

```python
import torch
import torch.nn as nn

class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # 2 inputs → 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden → 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))  # Hidden layer with sigmoid
        x = self.sigmoid(self.output(x))  # Output layer with sigmoid
        return x

model = XOR_MLP()
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Train with backpropagation + gradient descent (code omitted for brevity)
# After ~1000 epochs, the model learns XOR.
```

### The formula (now with meaning)

**Perceptron:**

z = Σᵢ wᵢxᵢ + b  
ŷ = activation(z)

**Common activation functions:**

1. **Sigmoid:** σ(z) = 1 / (1 + e^(-z))  
   Range: (0, 1). Smooth, but suffers from *vanishing gradients* (derivative → 0 for large |z|).

2. **Tanh:** tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))  
   Range: (-1, 1). Zero-centered (better than sigmoid), but still vanishing gradients.

3. **ReLU:** ReLU(z) = max(0, z)  
   Range: [0, ∞). Fast, no vanishing gradient for z > 0. Default choice for hidden layers. Problem: "dying ReLU" (neurons with z < 0 output 0 forever).

**Why ReLU beats sigmoid for deep networks:**

Sigmoid derivative: σ'(z) = σ(z)(1 - σ(z)) → max value = 0.25.

In a 10-layer network, gradient = 0.25^10 ≈ 0.00000095 → vanishing gradient. Weights do not update.

ReLU derivative: 1 if z > 0, else 0. No vanishing gradient for positive z. Gradients flow cleanly through deep networks.

### Common misunderstandings

- "More layers always means better performance." False. Without enough data, deep networks overfit. For tabular data with 50 features, a 2-layer MLP often beats a 10-layer network.
- "Sigmoid is obsolete." Not for the *output* layer of binary classification (you need probabilities 0–1). Sigmoid is bad for *hidden* layers (vanishing gradient), but fine for output.
- "ReLU is always best." Not for output layers where you need bounded output. Use ReLU for hidden layers, sigmoid/softmax for output.

### Exam angle

MCQs will ask: "Why can a perceptron not learn XOR?" Correct: "XOR is not linearly separable." Trap: "The perceptron has no bias term" (wrong — even with bias, it fails on XOR). Another question: "Why is ReLU preferred over sigmoid in deep networks?" Correct: "ReLU avoids vanishing gradients." Trap: "ReLU is faster to compute" (true, but not the *main* reason).

---

## Concept 10: Backpropagation and Gradient Descent

### The problem it solves

A neural network has millions of weights. How do you adjust each weight to reduce loss? Brute force (try every combination) is impossible. Backpropagation computes the gradient of loss with respect to *every* weight in one pass using the chain rule, enabling efficient learning.

### The intuition

Imagine you are blindfolded on a hill (loss surface) and want to walk downhill (minimize loss). You can feel the slope under your feet (gradient). Gradient descent says: take a step *opposite* to the slope. Repeat until you reach the bottom (local minimum).

**Forward pass:**  
Input → Layer 1 → Layer 2 → ... → Output → Loss.

You compute predictions and measure error (loss function).

**Backward pass (backpropagation):**  
Loss → Gradient of Layer N → Gradient of Layer N-1 → ... → Gradient of Layer 1.

Using the chain rule, you compute how much each weight contributed to the loss. Then adjust weights to reduce loss.

### Worked example — 2-layer network, 1 sample

**Setup:**

Input: x = 1.0  
Target: y = 1.0  
Weights: w₁ = 0.6, w₂ = 0.4, w₃ = 0.8, w₄ = 0.5  
Learning rate: η = 0.5

**Forward pass:**

Hidden neuron 1:  
z₁ = w₁ × x = 0.6 × 1.0 = 0.6  
h₁ = sigmoid(z₁) = 1 / (1 + e^(-0.6)) ≈ 0.6457

Hidden neuron 2:  
z₂ = w₂ × x = 0.4 × 1.0 = 0.4  
h₂ = sigmoid(z₂) ≈ 0.5987

Output:  
z₃ = w₃ × h₁ + w₄ × h₂ = 0.8 × 0.6457 + 0.5 × 0.5987 ≈ 0.8159  
ŷ = sigmoid(z₃) ≈ 0.6934

**Loss (MSE):**  
L = 0.5 × (y - ŷ)² = 0.5 × (1.0 - 0.6934)² ≈ 0.0470

**Backward pass:**

Output layer error:  
δ_out = -(y - ŷ) × ŷ × (1 - ŷ) = -0.3066 × 0.6934 × 0.3066 ≈ -0.0652

Gradient for w₃:  
∂L/∂w₃ = δ_out × h₁ = -0.0652 × 0.6457 ≈ -0.0421

Update w₃:  
w₃_new = w₃ - η × (∂L/∂w₃) = 0.8 - 0.5 × (-0.0421) = 0.8 + 0.0210 ≈ 0.8210

Repeat for w₄, w₁, w₂. After 1 epoch, loss drops from 0.0470 → 0.0454. Repeat for 1,000 epochs → loss → 0.

### The formula (now with meaning)

**Chain rule:**

∂L/∂w = (∂L/∂ŷ) × (∂ŷ/∂z) × (∂z/∂w)

Example: Loss depends on output ŷ, ŷ depends on z (weighted sum), z depends on w. Chain them together.

**Gradient descent update:**

w_new = w - η × (∂L/∂w)

where η (eta) is the learning rate (step size). Typical values: 0.001 to 0.1.

If η is too large, you overshoot the minimum and diverge. If η is too small, training is slow.

**Variants:**

- **SGD (Stochastic Gradient Descent):** Update weights after *each* sample. Fast but noisy.
- **Mini-batch GD:** Update after a batch of 32–256 samples. Balances speed and stability. Default in PyTorch.
- **Adam:** Adaptive learning rate. Combines momentum + RMSProp. Most popular optimizer.

### Common misunderstandings

- "Backpropagation is a different algorithm from gradient descent." False. Backpropagation *computes* gradients (using chain rule). Gradient descent *uses* those gradients to update weights. They work together.
- "Learning rate does not matter." False. If η = 10, you will overshoot and diverge. If η = 0.0001, training takes forever. Use Adam optimizer (auto-adjusts η) or a learning rate scheduler.
- "We always find the global minimum." False. Gradient descent finds a *local* minimum. For non-convex loss surfaces (neural networks), there are many local minima. Luckily, in practice, most local minima are good enough.

### Exam angle

MCQs will show a loss curve with train loss = 5.0 → 0.1 over 100 epochs, and ask "What happened?" Correct: "Gradient descent successfully minimized loss." Trap: "The model overfitted" (wrong — overfit means train loss low but test loss high; here we only see train loss decreasing).

Another question: "What is the learning rate's role?" Correct: "Controls step size during weight updates." Trap: "Determines the number of layers" (wrong — learning rate is a hyperparameter for optimization, not architecture).

---

## Concept 11: Overfitting, Underfitting, Bias-Variance Tradeoff

### The problem it solves

A model that memorizes training data fails on new data (overfitting). A model that is too simple fails on both (underfitting). Understanding the bias-variance tradeoff helps you diagnose which problem you have and how to fix it.

### The intuition

**Underfitting (high bias):**  
You are fitting a straight line to data that curves. Train error = high, test error = high. The model is too *simple* to capture the pattern.

**Good fit:**  
You fit a curve that captures the trend without memorizing noise. Train error = low, test error = low. This is the goal.

**Overfitting (high variance):**  
You fit a 10th-degree polynomial that passes through *every* training point, including noise. Train error = 0%, test error = 50%. The model memorized the training set.

**Bias-Variance Tradeoff:**

Total Error = Bias² + Variance + Irreducible Noise

- **Bias**: error from wrong assumptions (e.g., assuming data is linear when it is quadratic). High bias → underfitting.
- **Variance**: error from sensitivity to training data (e.g., model changes drastically if you add one sample). High variance → overfitting.

Goal: find the sweet spot where both are low.

### Worked example

**Dataset:** y = sin(x) + noise

Fit 3 models:

1. **Degree-1 polynomial (linear):** y = ax + b  
   Train error = 0.3, Test error = 0.3 → underfitting. The line cannot capture the sine curve.

2. **Degree-5 polynomial:**  
   Train error = 0.05, Test error = 0.06 → good fit. Captures the curve without memorizing noise.

3. **Degree-15 polynomial:**  
   Train error = 0.0, Test error = 0.5 → overfitting. Passes through every training point, but wiggles wildly between them.

**How to detect overfitting:**

Plot train loss vs. validation loss over epochs:

- Train loss ↓, Val loss ↓ → good.
- Train loss ↓, Val loss ↑ → overfitting. Stop training (early stopping).

### The formula (now with meaning)

**Bias:** Measures how wrong the model's *assumptions* are.

Bias = E[ŷ] - y_true

High bias → model cannot fit the data even with infinite training time.

**Variance:** Measures how much the model *changes* if you train on a different sample.

Variance = E[(ŷ - E[ŷ])²]

High variance → model is too sensitive to training data (memorizes noise).

**Total Error:**

E[(y - ŷ)²] = Bias² + Variance + σ²

where σ² = irreducible noise (the best model still has this error).

### Common misunderstandings

- "Overfitting means the model is too complex." Partially true. It also happens when you have *too little data* for a given model complexity. With infinite data, even a 100-layer network does not overfit.
- "High training accuracy means good model." False. If train accuracy = 100% but test accuracy = 50%, the model overfitted.
- "I should minimize variance only." False. If you use a constant model (ŷ = mean(y)), variance = 0 but bias is huge. You need to balance both.

### Exam angle

MCQs will show loss curves and ask "What is wrong?" Train loss = 0.01, Val loss = 0.5 → overfitting. Train loss = 0.3, Val loss = 0.3 → underfitting. The trap: "Val loss is higher than train loss" (this is *expected* — val loss is always slightly higher because the model has not seen val data. The problem is when the *gap* is huge).

---

## Concept 12: Regularization — L1, L2, Dropout, Early Stopping

### The problem it solves

Regularization prevents overfitting by penalizing model complexity. Without regularization, a neural network with 1 million parameters can memorize the training set. With regularization, the network is forced to learn *general* patterns instead of memorizing examples.

### The intuition

Think of regularization as a "simplicity tax." You want to minimize loss, but you also want to keep weights *small* (L2) or *sparse* (L1). Large weights mean the model is fitting to noise.

**L2 Regularization (Ridge):**  
Add a penalty term to the loss:

Loss_total = Loss_data + λ × Σ wᵢ²

where λ (lambda) controls the strength of the penalty. Large λ → strong penalty → weights shrink toward zero → simpler model.

Why does this help? Large weights mean the model is *certain* about patterns that might be noise. Penalizing large weights forces the model to be more "hesitant," improving generalization.

**L1 Regularization (Lasso):**  
Penalty = λ × Σ |wᵢ|

L1 drives some weights *exactly* to zero, performing automatic feature selection. Useful when you suspect many features are irrelevant.

**Dropout:**  
During training, randomly zero out 50% of neurons in each layer. Forces the network to learn *redundant* representations (if one neuron is dropped, others can compensate). Prevents co-adaptation (neurons relying too much on each other).

**Early Stopping:**  
Monitor validation loss. When it stops improving (e.g., 5 epochs without improvement), stop training. Saves the best checkpoint. Prevents overfitting without adding a penalty term.

### Worked example

**L2 regularization in sklearn:**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha = λ (penalty strength)
model.fit(X_train, y_train)
```

**Without regularization (alpha=0):**  
Weights: [10.2, -8.5, 0.1, 12.7, -9.3] → large magnitudes, overfitting.

**With regularization (alpha=1.0):**  
Weights: [2.1, -1.3, 0.05, 2.8, -1.9] → smaller magnitudes, better generalization.

**Dropout in PyTorch:**

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # Drop 50% of neurons during training
    nn.Linear(64, 10)
)
```

During training, half the neurons are randomly set to 0. During evaluation, all neurons are active (no dropout).

**Early stopping in Keras:**

```python
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if val loss does not improve for 5 epochs
    restore_best_weights=True
)

model.fit(X, y, validation_split=0.2, callbacks=[callback])
```

### The formula (now with meaning)

**L2 (Ridge):**

Loss_total = Loss_data + λ Σᵢ wᵢ²

Derivative: ∂Loss/∂w = ∂Loss_data/∂w + 2λw

Weight update: w_new = w - η(∂Loss_data/∂w + 2λw) = (1 - 2ηλ)w - η∂Loss_data/∂w

The term (1 - 2ηλ) *shrinks* w toward zero every step → "weight decay."

**L1 (Lasso):**

Loss_total = Loss_data + λ Σᵢ |wᵢ|

Derivative: ∂Loss/∂w = ∂Loss_data/∂w + λ × sign(w)

L1 subtracts a constant λ from the gradient, driving small weights *exactly* to zero (sparsity).

**Why L1 gives sparsity and L2 does not:**

L1 penalty contour is diamond-shaped. Loss contours are ellipses. The diamond's *corners* are on the axes (w₁=0 or w₂=0), so the minimum often lands on a corner → sparse solution.

L2 penalty contour is a circle. The minimum rarely lands exactly on an axis → no sparsity.

### Common misunderstandings

- "Regularization always improves performance." False. If your model is already underfitting (high bias), adding regularization makes it worse. Regularization helps when you are overfitting (high variance).
- "L1 and L2 do the same thing." False. L1 gives sparse weights (feature selection). L2 gives small but non-zero weights (all features contribute).
- "Dropout slows training." True, but it prevents overfitting, so you get better test accuracy with fewer epochs.

### Exam angle

MCQs will ask: "Model A has train error = 0.01, test error = 0.3. What should you do?" Correct: "Add regularization (L2, dropout) or get more data." Trap: "Increase model complexity" (wrong — this makes overfitting worse).

Another question: "What is the effect of increasing λ in Ridge regression?" Correct: "Weights shrink toward zero; model becomes simpler." Trap: "Training loss decreases" (wrong — training loss *increases* because you are penalizing complexity).

---

## Concept 13: Parameters vs Hyperparameters

### The problem it solves

Beginners confuse "learning rate" (hyperparameter) with "weight" (parameter). Understanding the distinction clarifies which values the model learns automatically and which values *you* must choose before training.

### The intuition

**Parameters:**  
Learned *automatically* during training by the optimizer. Internal to the model. Examples: weights, biases. You do not set these — the training loop adjusts them to minimize loss.

**Hyperparameters:**  
Set *manually* by you before training. External configuration. Examples: learning rate, number of layers, batch size, epochs. These control *how* the model learns, not *what* it learns.

Analogy: Building a house.
- **Parameters**: exact position of each brick (determined by the construction process).
- **Hyperparameters**: blueprint decisions (how many floors? what materials?). You decide these *before* construction starts.

### Worked example

```python
import torch
import torch.nn as nn

# Hyperparameters (you choose these):
learning_rate = 0.001
num_layers = 3
hidden_size = 128
batch_size = 32
num_epochs = 100

# Model architecture (hyperparameters determine structure):
model = nn.Sequential(
    nn.Linear(784, hidden_size),  # Parameters: weights and biases (learned)
    nn.ReLU(),
    nn.Linear(hidden_size, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop (optimizer adjusts parameters automatically):
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        loss = loss_function(model(X_batch), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Updates weights and biases (parameters)
```

**Parameters:** All the weights and biases in `nn.Linear` layers. These start as random values and get adjusted every step.

**Hyperparameters:** `learning_rate`, `num_layers`, `hidden_size`, `batch_size`, `num_epochs`. You set these before training and they stay fixed.

### The formula (now with meaning)

**Parameter count example:**

Fully connected layer: input_size → output_size.

Parameters = (input_size × output_size) + output_size (bias)

For a 784 → 128 layer:  
Parameters = (784 × 128) + 128 = 100,352 + 128 = 100,480

**Hyperparameter tuning:**

You search over a grid of hyperparameters and pick the best:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Hyperparameter
    'max_iter': [100, 1000]     # Hyperparameter
}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)  # {'alpha': 1.0, 'max_iter': 1000}
```

### Common misunderstandings

- "Learning rate is a parameter." False. The optimizer *uses* the learning rate to update parameters, but the learning rate itself is not learned — you set it.
- "I can tune parameters during training." False. Parameters are *automatically* tuned by backpropagation + optimizer. You tune *hyperparameters* by running multiple experiments.

### Exam angle

MCQs will list [weights, learning rate, batch size, biases, number of layers] and ask "Which are hyperparameters?" Correct: learning rate, batch size, number of layers. Trap: including "weights" or "biases" (those are parameters).

---

## Concept 14: PCA — Dimensionality Reduction

### The problem it solves

Real-world data lives in high dimensions (images = 150,000 pixels, genomics = 20,000 genes). Most dimensions contain noise or redundancy. PCA compresses data to a low-dimensional subspace (e.g., 2D for visualization) while preserving most variance (information).

### The intuition

Imagine you have a dataset of people's heights and weights. Height and weight are *correlated* (tall people tend to weigh more). PCA finds the "direction of maximum variance" — a diagonal axis that captures both height and weight in one number (roughly "body size").

**PC1** (first principal component) = direction of maximum variance (body size).  
**PC2** (second principal component) = direction of maximum remaining variance (body shape, orthogonal to PC1).

You can then represent each person using (PC1, PC2) instead of (height, weight), reducing 2D → 2D (no compression). But for 10,000 dimensions → 50 dimensions, PCA is hugely useful.

**Why does this work?**  
Real data has *structure*. Images of faces do not fill all 150,000 dimensions uniformly — most dimensions are "nose + mouth + eyes" variations. PCA finds the 50 dimensions that matter (facial features) and discards the 149,950 dimensions that are noise (pixel-level jitter).

### Worked example

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load handwritten digits (64 dimensions: 8×8 pixels)
digits = load_digits()
X = digits.data  # shape: (1797, 64)

# Reduce 64 dimensions → 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # shape: (1797, 2)

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='tab10', s=5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Digit')
plt.show()
```

**Result:** Digits cluster by class in 2D. PCA compressed 64D → 2D while preserving class structure.

**Explained variance:**

```python
print(pca.explained_variance_ratio_)
# [0.15, 0.13]  → PC1 captures 15% of variance, PC2 captures 13%
# Together: 28% of total variance in just 2 dimensions
```

To capture 95% of variance, you might need 20 principal components instead of all 64.

### The formula (now with meaning)

**PCA steps:**

1. Center the data: subtract mean from each feature.
2. Compute covariance matrix: C = (1/n) X^T X.
3. Compute eigenvectors and eigenvalues of C.
4. Sort eigenvectors by eigenvalue (largest = most variance).
5. Project data onto top-k eigenvectors.

**Projection:**

X_pca = X × V_k

where V_k = matrix of top-k eigenvectors.

**Explained variance ratio:**

ratio_i = λᵢ / Σⱼ λⱼ

where λᵢ = eigenvalue of i-th component.

### Common misunderstandings

- "PCA is supervised." False. PCA is unsupervised — it does not use labels. It only looks at the variance structure of X.
- "PCA always improves model accuracy." False. PCA *discards* information (low-variance dimensions). If those dimensions contain signal, accuracy drops. Use PCA for visualization or when training is too slow.
- "PCA gives interpretable features." Sometimes. For images, PC1 might be "brightness." But often, principal components are abstract linear combinations with no clear meaning.

### Exam angle

MCQs will ask: "Why use PCA?" Correct: "Reduce dimensionality while preserving variance." Trap: "Increase model accuracy" (PCA can help by reducing noise, but it does not *guarantee* better accuracy). Another trap: "PCA is used for feature scaling" (wrong — that is StandardScaler).

---

## Putting it together

This lecture builds a hierarchy:

1. **AI/ML/DL**: Understand the landscape (rule-based → data-driven → deep learning).
2. **Supervised vs Unsupervised**: Labels or no labels.
3. **Train-test split**: Detect overfitting by holding out data.
4. **Confusion matrix**: Break down where the model fails (TP, FP, FN, TN).
5. **Precision/Recall/F1**: Quantify failure modes and choose the right metric for the task.
6. **Feature scaling**: Make gradient descent converge faster.
7. **Sparse → Dense representations**: One-hot (orthogonal, wasteful) → embeddings (dense, meaningful).
8. **Distributional hypothesis**: Learn meaning from context (Firth 1957).
9. **Perceptron → MLP**: Single layer (linear) → multi-layer (non-linear).
10. **Backpropagation**: Compute gradients efficiently via chain rule.
11. **Overfitting/Regularization**: Diagnose via train-val curves; fix with L1/L2/dropout/early stopping.
12. **Parameters vs Hyperparameters**: Model learns parameters; you tune hyperparameters.
13. **PCA**: Compress high-dimensional data to low-dimensional subspace.

The through-line: modern ML is about *learning representations* from data. You start with raw features (pixels, words), the model learns better representations (edges, textures, word meanings), and you measure success with the right metric (F1 for imbalanced classes, recall for cancer detection). Overfitting is your enemy; regularization and train-test splits are your defense.

---

## Check yourself (5 questions)

**Q1.** A spam filter has the following confusion matrix on 1,000 test emails:

|                     | Predicted Spam | Predicted Ham |
|---------------------|----------------|---------------|
| **Actual Spam**     | 80             | 20            |
| **Actual Ham**      | 10             | 890           |

Which metric should the email provider optimize to avoid losing customer trust?

A. Recall — catch all spam, even if some ham is flagged.  
B. Precision — ensure flagged emails are actually spam.  
C. Accuracy — maximize overall correctness.  
D. F1 — balance both precision and recall equally.

**Answer:** B — In email, a False Positive (ham in spam folder) means the user misses important messages (job offer, meeting invite), which destroys trust. Precision = TP / (TP + FP) = 80 / 90 ≈ 88.9%. The provider wants precision > 99% so users trust the spam folder. Recall (catching all spam) is less critical — one spam in the inbox is annoying but not trust-breaking.

---

**Q2.** A neural network's training and validation loss curves over 50 epochs:

- Epoch 1: Train = 0.5, Val = 0.5
- Epoch 10: Train = 0.3, Val = 0.32
- Epoch 30: Train = 0.1, Val = 0.25
- Epoch 50: Train = 0.01, Val = 0.40

What is the problem, and what is the fix?

A. Underfitting. Fix: train longer (100 epochs).  
B. Overfitting. Fix: stop at epoch 10 (early stopping).  
C. Overfitting. Fix: add L2 regularization or dropout.  
D. Correct fit. No fix needed.

**Answer:** C — Train loss drops to 0.01 (model fits training data perfectly), but val loss rises from 0.25 → 0.40 (model fails on unseen data). This is overfitting. Early stopping at epoch 30 helps, but the question asks for a fix that lets training continue. L2 regularization penalizes large weights, and dropout prevents co-adaptation, both reducing overfitting. Option B is partially correct (early stopping), but C is better because it addresses the root cause.

---

**Q3.** Why does Word2Vec represent "cat" and "kitten" as similar vectors, while one-hot encoding represents them as orthogonal (dot product = 0)?

A. Word2Vec uses a larger embedding dimension (300 vs. 1).  
B. Word2Vec learns vectors by predicting context words; "cat" and "kitten" appear in similar contexts.  
C. Word2Vec uses a neural network, while one-hot encoding is linear algebra.  
D. One-hot encoding is sparse, so it cannot compute dot products.

**Answer:** B — The distributional hypothesis (Firth 1957) says words with similar contexts have similar meanings. Word2Vec trains on "predict context from target" (Skip-gram) or "predict target from context" (CBOW). Both "cat" and "kitten" appear near {pet, fur, meow, small}, so their learned vectors are close. One-hot vectors are fixed: each word is a basis vector, orthogonal by construction. Option A is wrong (dimension size is not the reason). Option C is wrong (the key is *learning from context*, not the network itself). Option D is wrong (you can compute dot products on sparse vectors; the issue is they are always zero).

---

**Q4.** You train a logistic regression model on a dataset where Feature 1 (income) ranges from 20,000 to 200,000 and Feature 2 (age) ranges from 18 to 80. Without feature scaling, gradient descent takes 10,000 iterations to converge. You apply StandardScaler. What happens?

A. The model's accuracy increases because scaling adds information.  
B. Gradient descent converges in ~100 iterations (100× faster), but final accuracy is the same.  
C. The model underfits because scaling removes variance.  
D. Scaling has no effect because logistic regression is scale-invariant.

**Answer:** B — Feature scaling does not change the model's expressive power (the decision boundary is the same), but it changes the *shape* of the loss surface. Before scaling, income dominates (range 180,000) while age barely matters (range 62), causing gradient descent to zigzag. After scaling (both features have mean=0, std=1), the loss surface is spherical, so gradient descent walks straight toward the minimum. Convergence speed improves 100×, but final accuracy is identical. Option A is wrong (scaling does not add information). Option C is wrong (scaling does not remove variance, just normalizes it). Option D is wrong (logistic regression *is* scale-sensitive because it uses gradient descent).

---

**Q5.** A binary classifier predicts cancer (positive class) or healthy (negative class). A doctor says, "I would rather have 10 false alarms than miss 1 cancer case." Which metric should the model optimize?

A. Precision — minimize false alarms.  
B. Recall — catch all cancer cases.  
C. F1 — balance both.  
D. Accuracy — overall correctness.

**Answer:** B — The doctor prioritizes minimizing False Negatives (missed cancer cases). Recall = TP / (TP + FN) measures "Of all actual cancer cases, how many did we catch?" High recall means low FN (few missed cases), even if FP (false alarms) is high. The doctor explicitly accepts 10 false alarms (FP) to avoid 1 missed case (FN), so recall is the right metric. Precision (option A) minimizes FP, which is the opposite of what the doctor wants. F1 (option C) balances both, but the doctor does not want balance — they want to prioritize recall. Accuracy (option D) is useless for imbalanced data (e.g., 1% cancer prevalence).

---

## Quick reference (for last-day revision)

**AI/ML/DL:**  
AI (goal) ⊃ ML (learn from data) ⊃ DL (multi-layer networks).

**Supervised:**  
Has labels. Classification (cat/dog) or Regression (price).

**Unsupervised:**  
No labels. Clustering (K-Means), Dimensionality Reduction (PCA).

**Train-Test Split:**  
80-20. Detect overfitting. Never tune on test set.

**Confusion Matrix:**  
TP, TN, FP, FN. Diagonal = correct. Off-diagonal = errors.

**Metrics:**  
- Accuracy = (TP+TN) / Total. Use for balanced data.  
- Precision = TP / (TP+FP). "Of flagged items, how many correct?"  
- Recall = TP / (TP+FN). "Of actual positives, how many caught?"  
- F1 = 2PR / (P+R). Harmonic mean. Use for imbalanced data.

**When to prioritize:**  
- Precision: spam filtering (avoid false alarms).  
- Recall: cancer detection (avoid missed cases).  
- F1: fraud detection (both errors costly).

**Feature Scaling:**  
StandardScaler: (x - μ) / σ → mean=0, std=1. Speeds up gradient descent. Fit on train, transform test.

**Sparse vs Dense:**  
- One-hot: [0,0,1,0,0]. Huge, orthogonal (dot product = 0).  
- Embedding: [0.3, -0.1, 0.8]. Compact, similar words close.

**Distributional Hypothesis:**  
"Know a word by its context" (Firth 1957). Word2Vec learns embeddings.

**Perceptron → MLP:**  
Perceptron = 1 layer, linear. MLP = multi-layer, non-linear (via activation).

**Activation Functions:**  
- Sigmoid: (0,1). Output layer for binary classification.  
- ReLU: max(0,z). Hidden layers (no vanishing gradient).  
- Tanh: (-1,1). Zero-centered.

**Backpropagation:**  
Compute ∂L/∂w via chain rule. Update: w ← w - η∂L/∂w.

**Overfitting:**  
Train loss ↓, Val loss ↑. Fix: L2, dropout, early stopping, more data.

**Regularization:**  
- L2 (Ridge): Loss + λΣw². Shrinks weights.  
- L1 (Lasso): Loss + λΣ|w|. Sparsity (feature selection).  
- Dropout: Randomly zero neurons (p=0.5). Prevents co-adaptation.  
- Early Stopping: Stop when val loss stops improving.

**Parameters vs Hyperparameters:**  
Parameters = learned (weights). Hyperparameters = set by you (learning rate, layers).

**PCA:**  
Unsupervised dimensionality reduction. Finds axes of max variance. X_pca = X × V_k.

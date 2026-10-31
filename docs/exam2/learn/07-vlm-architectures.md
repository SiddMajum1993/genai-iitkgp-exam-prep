# 07 · VLM architectures — CLIP, LLaVA, Qwen-VL and SAM

**Session:** 29 August 2026 · Faculty not named in the course files · Module 4, Lecture 1, *Vision-Language Model Architectures*
**Read time:** 40–50 min · **Builds on:** transformers and attention from the 1st exam; Chapter 03 (adapters) and Chapter 04 (LoRA) for the "frozen backbone + small trainable part" idea

---

## In one minute

A **vision-language model (VLM)** takes an image (plus some text) and produces text, a label, a similarity score or a mask. This lecture covered four famous models, and each one stands for a *different way* of combining vision and language:

**CLIP matches.** It places images and text in one shared space, and its output is a similarity score or a label.

**LLaVA describes and chats.** You show it an image and talk about it, and it replies in free-form text.

**Qwen-VL chats, but can also point and read.** On top of conversation it does grounding and OCR, so its output is text plus bounding boxes.

**SAM cuts out.** You point at an object and it returns exactly that object as a pixel mask.

They all share one ingredient: a **Vision Transformer (ViT)** that turns an image into a sequence of patch tokens. What differs is **the connector** (how vision meets language) and **what the output is used for**.

If you remember one sentence, make it this one: **same eyes (ViT), different jobs: match, describe, ground, segment.**

---

## Exam radar

How likely each topic is to be tested, based on slide time, the in-class quick-check quizzes, the 39-question Q&A sheet and whether it came back in the next lecture (Chapter 08).

### 🔴 High: expect questions here

**CLIP's contrastive training.** The image–text grid, positives vs negatives, temperature and zero-shot classification. It got the heaviest slide time, a 6-question quick check and about 12 Q&A questions.

**LLaVA, and what is trained in Stage 1 vs Stage 2.** LLaVA is a frozen CLIP encoder, a projector and an LLM. It had a 6-question quick check, and "which part is frozen" is a classic question.

**Qwen-VL's cross-attention adapter with 256 learned queries.** Also its grounding tokens and three training stages. One of the lecture's stated learning objectives was to "trace how a pixel becomes a token".

**Choosing the paradigm and "where fusion happens".** Contrastive vs generative vs promptable. This was the anchor comparison of the lecture and is easy material for scenario questions.

### 🟠 Medium–High

**SAM.** A heavy encoder that runs once, a light decoder per prompt, 3 candidate masks, the prompt types and the data engine. It had a 6-question quick check and was taught again in the next lecture.

### 🟠 Medium: likely to show up in one question

**ViT basics: patches, positional embeddings and [CLS].** Several Q&A questions were about these.

**LLaVA's GPT-4-generated data, and the LLaVA-1.5 and LLaVA-NeXT upgrades.** These appeared in the quick-check items.

**The limitations of each model.** Expect "why does it fail?" scenario questions.

### 🟢 Low: understand the idea, skip the details

**Lists of model variants, exact parameter counts and history dates.** The professor said there was "no need to remember" these.

---

## The big picture: one template, four variations

The professor drew one generic diagram and said *every model is a variation on it*:

```mermaid
flowchart LR
    I[Image] --> VE["Vision encoder<br/>(ViT or CNN)"]
    T[Text] --> TE["Text encoder<br/>or tokenizer"]
    VE --> C["Connector / fusion"]
    TE --> C
    C --> O["Output:<br/>score · text · boxes · mask"]
```

The most useful lens is **where the two modalities meet** ("where fusion happens"):

**No fusion (dual encoder).** There are two separate towers, one for images and one for text, and they only "meet" in a final similarity score. This is CLIP.

**Early fusion into an LLM.** Image tokens are fed into the LLM alongside the words, and ordinary self-attention mixes them. This is LLaVA and Qwen-VL.

**A cross-attention module.** A dedicated attention block lets one set of tokens look at another. Qwen-VL's adapter and SAM's mask decoder both work this way.

On top of that, the models fall into three **paradigms**, depending on what goes in and what comes out.

**Contrastive.** You give it an image plus candidate text labels, and it gives back similarity scores. Example: CLIP.

**Generative / instruction-following.** You give it an image plus a free-text instruction, and it generates text. Examples: LLaVA and Qwen-VL.

**Promptable dense prediction.** You give it an image plus a spatial prompt (a click or a box), and it returns a pixel mask. Example: SAM.

---

## Concept 1 — The Vision Transformer (ViT): how a picture becomes tokens

### The idea in plain words

A text transformer reads a sentence as a sequence of word tokens. A ViT does the same for images:

1. **Cut the image into small square patches**, for example 16 × 16 pixels each.
2. **Flatten each patch into a vector** and pass it through a learned linear layer, just like a word embedding.
3. **Add a positional embedding** to each patch. Attention on its own doesn't know order, so this tells the model *where* each patch came from.
4. Optionally add a special **[CLS] token** whose final output summarises the whole image.
5. Run the sequence through ordinary **transformer encoder layers**.

The standard example: a 224 × 224 image with 16 × 16 patches gives **196 patch tokens** (197 with [CLS]). The formula is N = (H ÷ P) × (W ÷ P): (224 ÷ 16)² = 14² = 196. With 14 × 14 patches the same image gives 16² = **256**, and a 336 × 336 image gives 24² = **576**. Halving the patch size quadruples the count.

!!! example "Everyday analogy"
    Cut a photo into a grid of jigsaw pieces, write a "description card" for each piece and number the cards by position. The ViT reads the cards in order, like words in a sentence.

### Why it matters

CLIP, LLaVA's vision tower, Qwen-VL's encoder and SAM's encoder are **all ViT variants**. Learn it once and it explains all four models.

### Common misunderstandings

- **"The positional encoding is just the numbers 1, 2, 3…"** No. Each position gets a *vector* that is added to the patch vector.
- **"A ViT reads text in an image like OCR software."** No. It sees text as visual patterns. Reading ability comes from training that links those patterns to language.
- **"ViT always beats CNNs."** Not always. The Q&A answer: a **ResNet (CNN) can be better on smaller datasets** or when efficiency matters. ViT shines with lots of data and compute.

---

## Concept 2 — CLIP: teaching images and captions to meet in one space

**CLIP** (OpenAI, 2021) stands for *Contrastive Language–Image Pre-training*.

### The problem it solves

A classic image classifier (like a ResNet trained on ImageNet) knows a **fixed list of about 1,000 classes**. To add a new class, you collect labelled photos and retrain. It can't recognise anything outside its list.

### The idea in plain words

Instead of a fixed list of labels, learn from **400 million (image, caption) pairs** scraped from the web (the WIT dataset). Train two separate encoders:

- an **image encoder** (a ViT or ResNet) that turns an image into a vector, and
- a **text encoder** (a transformer) that turns a caption into a vector.

Both vectors land in the **same shared space**. Training pulls each image *close to its own caption* and pushes it *away from every other caption*. That's what **contrastive** means.

The professor's framing: *treat labels as text, put them as points in the same space as images, and classification becomes a nearest-neighbour search.*

!!! example "Everyday analogy"
    A party where every guest (an image) has a matching name badge (a caption) mixed into a pile. The game: each guest must find their own badge and stay far away from everyone else's. After enough rounds, guests and badges that belong together always end up standing side by side.

### How one training step works (the grid picture)

Take a batch of N image–caption pairs. Compare **every image with every caption**, which makes an N × N grid of similarity scores.

Picture a batch of three images and their three captions: a photo of dogs with "two dogs running across a frosty field", a photo of a whale with "whale fin above the ocean", and a photo of a forest with "dirt path in a pine forest".

CLIP scores **every image against every caption**, so three images and three captions give nine pairings. Only three of them are true ✅: image 1 with caption 1, image 2 with caption 2, and image 3 with caption 3. The other six, like the dog photo with the whale caption, are mismatches ✖.

The slides draw these nine pairings as a square grid, with images down the side and captions along the top. The true pairs are the ones where the image number equals the caption number, so they run corner to corner. That's why the course calls them the **diagonal**.

- The **diagonal** cells are the true pairs (**positives**).
- Every **off-diagonal** cell is a wrong pair (a **negative**). For each image, the other N − 1 captions in the batch are its negatives, so **no one has to hand-pick negatives**.
- The loss is called **InfoNCE**, and CLIP uses a **symmetric** version. Each row asks "which caption matches this image?" and each column asks "which image matches this caption?". Both are treated as classification problems and the two losses are averaged.

**Bigger batches = more negatives = better learning.** CLIP used a huge batch of 32,768 pairs.

### Two small details that get asked

**Embeddings are L2-normalised.** Every vector is scaled to length 1, so the dot product equals the **cosine similarity**.

**The temperature τ (tau) is learned.** It rescales the scores before the softmax. **Lower τ = sharper, more confident choices; higher τ = softer.** CLIP *learns* τ instead of fixing it. For example, two captions with cosine scores 0.30 and 0.25 are nearly a coin flip at τ = 1 (about 51% vs 49%), but at τ = 0.01 they become 30 vs 25, and the first caption gets about **99%**.

**Projection layers make the sizes match.** The image and text encoders output different-sized vectors. Small learned projection layers map both into the same shared size.

**The text embedding is taken at the [EOS] token.** The caption's vector is the text encoder's output at the end-of-sequence token, not an average over all tokens.

**The modality gap.** Even after training, image embeddings and text embeddings don't fully mix. They sit in **two separate regions ("cones")** of the shared space, roughly parallel to each other. Retrieval and zero-shot classification still work, because what matters is which caption is *relatively* closest to an image, not the absolute distance. *(This comes from the official practice set and later research, beyond the course slides.)*

### Common misunderstandings

- **"CLIP is a Siamese network."** Related idea, but a Siamese network *shares one encoder*. CLIP has **two separate** encoders, one per modality.
- **"An accidentally mismatched caption in the data is a negative."** No. Negatives are the *intentional* wrong pairings inside the batch. A wrong caption in the data is **label noise**.
- **"You could just use PCA to align image and text vectors."** No. Alignment has to be *learned* through the contrastive objective.
- **"The shared space is universal."** No. It's specific to that trained model.

---

## Concept 3 — Zero-shot classification with CLIP

### The idea

**Zero-shot** means doing a task **without any training examples for that task**. With CLIP, you classify an image into classes it was never explicitly trained on:

1. Embed the image once.
2. Turn each class name into a sentence such as **"a photo of a {label}"** and embed each one with the text encoder.
3. Compute the similarity between the image and each class sentence.
4. A softmax turns the similarities into probabilities, and the highest one wins.

**Prompt wording matters.** Averaging over about 80 hand-written templates ("a photo of a big {label}", "a sketch of a {label}") improves accuracy.

!!! example "Small example"
    Classes: *cat, dog, giraffe.* You embed "a photo of a cat", "a photo of a dog" and "a photo of a giraffe". A picture of a giraffe lands closest to the third sentence, so CLIP says "giraffe", even if nobody ever trained it on a labelled "giraffe" class.

### Zero-shot vs linear probe

**A linear probe needs labels.** You freeze the encoder and train a small classifier on labelled target data.

**Zero-shot needs no labels.** There is no training at all. The class names are simply embedded as text.

The headline result: zero-shot CLIP reached **76.2% top-1 on ImageNet**, matching a fully supervised ResNet-50. It was also much more robust on shifted data such as sketches.

### Where CLIP is weak

- Fine detail: **counting**, reading small text (OCR) and **spatial relations** ("left of", "behind").
- Sensitive to prompt wording.
- Inherits web biases.
- **It can't generate text.** It only scores matches. That gap is exactly what LLaVA fills.

### Common misunderstanding

- **"Zero-shot = unsupervised."** No. CLIP was trained with (noisy) supervision from captions. "Zero-shot" only means no examples *for the new task*.

---

## Concept 4 — LLaVA: plugging an image into a chatbot

**LLaVA** = *Large Language and Vision Assistant* (2023). The two Ls stand for "Large Language".

### The problem it solves

CLIP can match but can't describe, explain or answer questions. GPT-4V showed multimodal chat was possible, but it was closed. LLaVA's question was: can we reuse an existing vision encoder and an existing LLM, and just teach them to work together cheaply?

### The idea in plain words

**"Treat the image as a foreign language."**

```mermaid
flowchart LR
    IMG[Image] --> CLIPV["CLIP ViT-L/14<br/>vision encoder ❄️ frozen"]
    CLIPV --> P["Projector 🔥<br/>(linear → 2-layer MLP in v1.5)"]
    P --> VT["Visual tokens<br/>(e.g. 256)"]
    TXT[Instruction text] --> TT[Text tokens]
    VT --> LLM["LLM (Vicuna / LLaMA)"]
    TT --> LLM
    LLM --> R[Text answer]
```

1. A **frozen CLIP vision encoder** turns the image into patch features.
2. A small **projector** translates those features into vectors that look like the LLM's word embeddings. LLaVA v1 used a single linear layer; **LLaVA-1.5 uses a 2-layer MLP**, which adds non-linear capacity and improves accuracy.
3. The resulting **visual tokens** are placed in the input sequence next to the text tokens: `[system prompt][visual tokens][user question][assistant answer]`.
4. The **LLM** reads everything with its normal self-attention and writes the answer. There's no separate cross-attention module.

### Why the projector is needed

Even if the CLIP vector and the LLM's word vectors have the same size, they live in **unrelated "geometries"**. Without the projector, a dog-image vector would be a random point to the LLM. The projector learns *where* that vector should land, a bit like a rotation and shift between two coordinate systems.

!!! example "Everyday analogy"
    The CLIP encoder is a photographer who speaks only Japanese; the LLM is a writer who speaks only English. The projector is a small, cheap interpreter between them. You don't retrain the photographer or replace the writer. You just hire the interpreter.

### Where the training data came from

No dataset of (image, instruction, answer) existed. The trick: take COCO image **captions and bounding boxes**, write them out as text, and ask a **text-only GPT-4** (it never sees the image) to create Q&A. This produced about **158K examples** of three types: **conversation**, **detailed description** and **complex reasoning**.

### The two training stages (very likely to be asked)

**Stage 1 · Feature alignment.** The goal is to teach the projector to map image features into the LLM's word space. The vision encoder is ❄️ frozen, the projector is 🔥 trained and the LLM is ❄️ frozen. The data is about 595K image–caption pairs.

**Stage 2 · Instruction tuning.** The goal is to teach the whole system to follow instructions and chat. The vision encoder stays ❄️ frozen, the projector is 🔥 trained and the LLM is now 🔥 trained too. The data is about 158K GPT-4-generated samples.

- The loss is ordinary next-token prediction, computed **only on the assistant's answer tokens**.
- The projector carries over from Stage 1 into Stage 2. It isn't reset.
- **Stage 2 is still needed even if the LLM was already instruction-tuned on text**, because it has never learned to interpret visual tokens.

### Upgrades worth recognising

- **LLaVA-1.5:** MLP projector, academic VQA data mixed in, and higher input resolution (336 px).
- **LLaVA-NeXT (1.6), "AnyRes":** split a high-resolution image into tiles, encode each tile, and add a downsized global view. This is better for small text, charts and documents, at the cost of more visual tokens.

### Limitations

Hallucination, small text lost at low resolution, weak counting and localisation, and it inherits CLIP's blind spots because the encoder is frozen.

### Common misunderstandings

- **"LLaVA can output images or masks."** The original outputs **text only**. Masks need another model, such as SAM.
- **"The projector is the expensive part."** No. Cost is driven by **the LLM and the number of visual tokens**. To cut cost, use a smaller or quantised LLM, fewer visual tokens or a lower resolution, and keep the encoder frozen.

---

## Concept 5 — Qwen-VL: a fixed-size summary that remembers *where*

**Qwen-VL** (Alibaba, 2023) targets what LLaVA is weak at: **grounding** (pointing at regions), **OCR** (reading dense text), and **multi-image / multi-round** chat.

### The problem it solves

At higher resolution (448 × 448), the ViT produces **hundreds to over a thousand** patch tokens per image. Feeding all of them into the LLM is expensive, especially with several images in one conversation.

### The idea: the position-aware adapter

Between the ViT and the LLM, Qwen-VL puts **a single cross-attention layer** with **256 learnable "query" vectors**:

- Each of the 256 queries *looks at all the patch features* (cross-attention) and produces one output token.
- **2D positional encodings** are added so the summary still knows *where* things were in the image.
- Whatever the image resolution or patch count, **exactly 256 visual tokens** come out.

```mermaid
flowchart LR
    P["ViT patch features<br/>(count varies with resolution)"] -->|"keys & values<br/>+ 2D positions"| CA["Cross-attention adapter"]
    Q["256 learnable queries 🔥"] -->|queries| CA
    CA --> OUT["Always 256 visual tokens"] --> LLM[Qwen LLM]
```

!!! example "Everyday analogy"
    A news editor (the adapter) with **256 reporters** (the queries). The raw footage might be 200 or 2,000 clips depending on the event. Each reporter watches *all* the footage and files **one** summary, tagged with *where* it happened. The newsroom (the LLM) always receives exactly 256 reports.

### LLaVA vs Qwen-VL connector

**The connector.** LLaVA uses a linear / MLP projector that keeps every patch. Qwen-VL uses a cross-attention adapter with 256 queries.

**The visual token count.** In LLaVA it grows with resolution or tiles. In Qwen-VL it is **fixed** at 256 regardless of resolution.

**Grounding.** LLaVA doesn't have it built in. Qwen-VL has it **natively**, via special tokens.

**Multiple images.** Support is limited in the base LLaVA model. Qwen-VL was designed for it.

**Training stages.** LLaVA has 2. Qwen-VL has 3.

### Grounding with special tokens

Qwen-VL emits boxes as **plain text inside special tags**, using normal next-token prediction. There's no special output head.

- `<ref>…</ref>` wraps the phrase being referred to.
- `<box>…</box>` wraps coordinates, written as numbers normalised to **0–1000**.
- `<img>…</img>` marks where an image's tokens go. Several `<img>` blocks allow multiple images ("Picture 1", "Picture 2").

Example: *"Find the red umbrella."* → `<ref>the red umbrella</ref><box>(342,128),(410,256)</box>`

The model learns to write these coordinates from **region-annotated training data**. Positional embeddings help, but they don't directly turn into boxes.

### Three training stages

1. **Pre-training, at 224 px.** The vision encoder and adapter are trained, and the **LLM is frozen**. The data is about 1.4B weakly labelled web image–text pairs.
2. **Multi-task pre-training, at 448 px.** The **whole model** is trained, on seven tasks together: captioning, VQA, grounding, OCR and more.
3. **Supervised instruction fine-tuning, at 448 px.** The LLM and adapter are trained, and the vision encoder is frozen again. The data is chat, multi-image and multi-round conversations, and the result is **Qwen-VL-Chat**.

### Limitations

The fixed 256 tokens can **lose fine detail** in very dense images such as large tables. The 3-stage recipe is costlier and harder to reproduce. Qwen2-VL later moved to **dynamic resolution**.

### Common misunderstanding

- **"256 tokens means the image is resized to 256 pixels."** No. It's 256 *summary tokens*, produced by attention over all patches at whatever resolution the encoder used.

---

## Concept 6 — SAM: cut out whatever you point at

**SAM** = *Segment Anything Model* (Meta AI, 2023).

### The problem it solves

Language had prompt-driven foundation models; segmentation didn't. Old segmentation models were trained per dataset and had to be retrained for new domains. SAM's task is **promptable segmentation**: given an image and a prompt (a click, a box or a rough mask), return a valid mask, even for objects and domains it never saw (**zero-shot transfer**).

### The three parts

**The image encoder is heavy.** It's a big ViT-H, MAE-pretrained. It turns the image into an embedding **once per image** and caches it. This takes about 0.15 s and is **the most expensive step**.

**The prompt encoder is light.** It encodes clicks and boxes (with positional encodings + learned type embeddings), or a rough mask (with convolutions). Text is only experimental.

**The mask decoder is light too.** It's a 2-layer transformer with **two-way attention**: prompt tokens look at image tokens *and* image tokens look at prompt tokens, and then it outputs masks. It takes about 50 ms per prompt.

**Why it feels instant:** the heavy encoder runs once, and every new click only re-runs the cheap decoder.

!!! example "Everyday analogy"
    A tailor who takes your full measurements once (the image encoder). After that, each alteration you ask for, like "shorten this sleeve" or "take in the waist" (the prompts), is quick, because the tailor doesn't re-measure you every time.

### Handling ambiguity: three masks

One click on a shirt button could mean *the button*, *the shirt* or *the whole person*. So SAM returns **3 candidate masks** (sub-part, part, whole), and a small **IoU-prediction head** scores each one so the best can be picked.

### Prompt types

**A point** (a foreground or background click) is good for fast interactive selection.

**A box** is good for precisely selecting one object.

**A rough mask** is good for refining an existing mask.

**Text** is experimental, and uses a CLIP text embedding.

**Not a prompt:** audio. (This was a quick-check question.)

### The data engine (how SA-1B was built)

1. **Assisted-manual:** annotators draw masks with SAM's help, and SAM is retrained periodically.
2. **Semi-automatic:** SAM proposes confident masks; annotators add what it missed.
3. **Fully automatic:** SAM is prompted with a **regular grid of points (32 × 32)**, keeps confident and stable masks, and removes near-duplicates.

The result, **SA-1B**, has about **1.1 billion masks on 11 million images**. It's a loop: *model annotates → new data trains model → better model annotates*.

### Where SAM stops

SAM knows **"where"**, not **"what"**. It gives a precise outline but doesn't name the object or answer questions. To get "find the red umbrella and cut it out", you **compose**: a grounding model finds and names it (box), then SAM turns the box into a pixel mask. This combination is known as **Grounded-SAM**. **SAM 2** extends SAM to video using a memory of previous frames.

### Common misunderstandings

- **"The prompt is a question like 'what is in this picture?'"** No. A prompt is a way of *pointing* at what to cut out.
- **"The mask decoder is the slow part."** No. The **image encoder** is.
- **"SAM outputs a cropped or zoomed image."** No. It outputs a mask laid over the same image.

---

## Putting it together: the four models side by side

**CLIP** (OpenAI, 2021) is contrastive. Its vision backbone is a ResNet or ViT trained from scratch. There's no fusion, only a shared embedding space. It outputs a similarity or a label, and it was trained on 400M web pairs.

**LLaVA** (Wisconsin / Microsoft, 2023) is generative (chat). Its vision backbone is a **frozen** CLIP ViT-L/14, connected through a linear / MLP projector. It outputs free text, and its training signal was 158K GPT-4 instructions.

**Qwen-VL** (Alibaba, 2023) is generative + grounded. Its vision backbone is a ViT initialised from OpenCLIP, connected through a cross-attention adapter (256 tokens). It outputs text + boxes, and it was trained on 1.4B pairs + multi-task data + instructions.

**SAM** (Meta AI, 2023) does promptable segmentation. Its vision backbone is an MAE-pretrained ViT-H, and its fusion happens in a two-way attention decoder. It outputs a mask + a quality score, and it was trained on 1.1B masks from a data engine.

The professor's summary: *"What differs is not how we see, but what we do with what we saw: match, describe, ground, segment."*

### Which one when?

**For image search, tagging, zero-shot labels or content moderation, choose CLIP.** It gives fast similarity, and no generation is needed.

**For a conversational description or general visual Q&A, choose a LLaVA-style model.** A frozen encoder + an LLM = chat.

**For pointing at regions, reading text, documents or multiple images, choose Qwen-VL.** It has grounding tokens, OCR training and a fixed token budget.

**For the exact pixel outline of an object, choose SAM**, often with a detector or VLM for the "what". It gives promptable, ambiguity-aware masks.

**To add grounding to a LLaVA-like model with minimal change, emit coordinates as text tokens, Qwen-VL style.** No new output head is needed.

### Design trade-offs to recognise

- **Frozen backbone** (LLaVA: cheap) vs **fine-tuned backbone** (Qwen-VL: more capable, costlier).
- **Token efficiency** (Qwen-VL's fixed 256) vs **spatial detail** (LLaVA-NeXT tiling keeps more patches).
- **Huge noisy web data** (CLIP, Qwen-VL) vs **curated or human-in-the-loop data** (LLaVA, SAM).
- **Specialist** (SAM) vs **generalist** (Qwen-VL).

---

## What the hands-on showed

!!! warning "Inferred"
    The hands-on for this session was shared only as a short link, and no notebook content is in the course files. The Q&A refers to Hugging Face TRL's `SFTTrainer` and a LoRA hands-on from the previous day. That suggests fine-tuning a VLM with LoRA, but we couldn't confirm the notebook's exact contents.

---

## Where students got confused (from the live Q&A)

These are real questions students asked during the session, with the answers they were given.

**"How many tokens does a 224 × 224 image give with 16 × 16 patches?"** 196 patch tokens (plus 1 if there's a [CLS] token).

**"Is the positional encoding just a number per patch?"** No. It's a *vector* per position, added to the patch vector.

**"Why does ViT need [CLS] but GPT doesn't?"** ViT needs one global image summary. A decoder-only GPT produces outputs token by token.

**"Is CLIP a Siamese network?"** Related idea, but Siamese networks share one encoder. CLIP has two separate ones.

**"Encoders give different-sized vectors. How do they match?"** Learned projection layers map both into the same shared space.

**"What does temperature do?"** Lower τ = sharper softmax; higher τ = softer.

**"Is a wrongly captioned web image a negative?"** No. That's label noise. Negatives are the other pairs in the batch.

**"Can PCA align image and text spaces?"** No. Alignment must be learned with the contrastive objective.

**"Is zero-shot the same as unsupervised learning?"** No. Zero-shot means no task-specific examples; training was still supervised by captions.

**"Can LLaVA output a mask or an image?"** The original LLaVA outputs text only.

**"Why is LLaVA Stage 2 needed if the LLM is already instruction-tuned?"** The LLM has never learned to interpret visual tokens.

**"Is the projector reset between stages?"** No. It's trained in both stages.

**"How do I make LLaVA cheaper to train?"** Use a smaller or quantised LLM and fewer visual tokens, and keep the encoder frozen.

**"How does Qwen-VL output boxes?"** It writes coordinates as text inside `<box>` tags, learned from region-annotated data.

---

## Traps to watch for in MCQs

- ❌ "CLIP generates captions." → It only **scores** image–text similarity.
- ❌ "CLIP needs manually mined negatives." → Negatives come free from the **other pairs in the batch**.
- ❌ "Higher temperature makes CLIP's predictions sharper." → **Lower** τ is sharper.
- ❌ "In LLaVA Stage 1, the LLM is fine-tuned." → Stage 1 trains **only the projector**.
- ❌ "In LLaVA Stage 2, the vision encoder is updated." → It stays **frozen**. Projector and LLM are trained.
- ❌ "LLaVA uses cross-attention to fuse the image." → It uses a **projector** and ordinary self-attention inside the LLM. Cross-attention is Qwen-VL's adapter.
- ❌ "Qwen-VL's visual token count grows with image resolution." → It's **fixed at 256**.
- ❌ "Qwen-VL predicts boxes with a detection head." → Boxes are **text tokens** inside `<box>` tags.
- ❌ "SAM tells you what the object is." → SAM tells you **where** (a mask), not what.
- ❌ "SAM re-runs its image encoder for every click." → The encoder runs **once**; only the decoder re-runs.
- ❌ "SAM returns one mask per prompt." → **Three**, scored by an IoU head.

---

## Recap

1. **One template:** ViT eyes → connector → output. The four models differ in the connector and the output.
2. **CLIP** = two encoders, one shared space. Contrastive training on an image–caption grid gives **zero-shot classification** via "a photo of a {label}". It can't generate text.
3. **LLaVA** = frozen CLIP encoder + **projector** + LLM. **Stage 1 trains only the projector; Stage 2 trains projector + LLM.** The encoder is always frozen.
4. **Qwen-VL** = cross-attention adapter with **256 learned queries** (fixed token count, positions kept) + `<ref>`/`<box>` grounding tokens + 3 training stages. Strong at OCR, grounding and multiple images.
5. **SAM** = heavy encoder **once**, light decoder **per prompt**, **3 candidate masks**, built on a self-improving **data engine** (SA-1B). It knows "where", not "what".

---

## Practice questions

Pick an option to see instantly whether you're right, with an explanation for every option.

<div class="practice-quiz" data-quiz="ch07">
  <p><em>Loading practice questions…</em></p>
</div>

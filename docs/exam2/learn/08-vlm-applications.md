# 08 · Applications of VLMs — picking the right model and the right prompt

**Session:** 5 September 2026 · Faculty not named in the course files · Module 4, Lecture 2, *Applications of VLM Architectures*
**Read time:** 30–40 min · **Builds on:** Chapter 07 (CLIP, LLaVA, Qwen-VL, SAM)

---

## In one minute

Chapter 07 gave you four tools. This lecture is the **toolbox manual**: for a real task, *which model* do you pick, *how* do you prompt it, and *what usually goes wrong*?

Every application comes down to **two stacked decisions**:

1. **Which model fits the task's shape?** Tag or label → CLIP. Free-text answer → LLaVA-style. Reading text, pointing or documents → Qwen-VL. Pixel outline → SAM.
2. **Which prompting technique?** Scoped zero-shot prompts → chain-of-thought (CoT) for counting and reasoning → an explicit JSON schema for extraction.

The five application areas: **image captioning, visual question answering (VQA), document intelligence, visual reasoning, and chart QA.** Each has a typical failure mode you should plan for.

If you remember one sentence, make it this one: **match the task's shape to the model, match the prompt to the task, and plan for the known failure.**

---

## Exam radar

How likely each topic is to be tested, based on slide time, the quick-check quizzes in each section, the recap table and the Q&A sheet.

### 🔴 High: expect questions here

**The rules for matching a task to a model.** A sentence calls for a different model than a tag. Qwen-VL is the choice for OCR, grounding and documents. Reasoning comes from the LLM. This is the premise of the whole lecture, and perfect material for scenario questions.

**How to prompt for each task.** Captions are scoped by length, counting and reasoning use chain-of-thought (CoT), extraction uses a JSON schema, and the prompt allows "not present". This was repeated in every quick check, the recap table and the takeaways.

**Failure modes.** Hallucination, presupposition hallucination, compounding errors, hallucinated fields, and the need for human review. Every section has a pitfalls slide and a quiz item on it.

### 🟠 Medium: likely to show up in one question

**The SAM recap (🟠 Medium–High).** The encoder runs once, there are 3 masks, several prompt types, the focal + Dice loss and the data engine. It was taught again with a new slide on the loss, and it drew most of the Q&A questions.

**Composing models.** A VLM finds the object and SAM cuts it out, and extraction pipelines differ from matching pipelines. Several long Q&A answers cover this.

**Datasets and metrics.** COCO, VQAv2, DocVQA/ANLS, MMMU, CIDEr, BLEU and CLIPScore. They're listed on the slides, so matching-style questions are possible.

**How state-of-the-art models are trained for each task.** For documents, resolution is the main lever. There's one slide per task.

**The DAVE and DocVLM case studies.** The slides mark them optional, but the official practice set asks three questions on them: DAVE's raw-pixel MAE target, DAVE's learned merging and DocVLM's 64 queries.

### 🟢 Low: understand the idea, skip the details

**Benchmark scores.** The numbers are only "directional".

---

## The application map

The lecture's own summary slide covers five applications. For each one, know the task, the model to reach for and how to prompt it.

**Image understanding and captioning: describe what's in the image.** Use a LLaVA-style model, or CLIP if you only need tags. Prompt zero-shot, and keep the prompt scoped and length-constrained.

**Visual Q&A: answer a specific question about the image.** Use a LLaVA-style model or Qwen-VL. Prompt zero-shot, or use CoT for counting and comparison.

**Document intelligence: extract structured fields from documents.** Use **Qwen-VL**. Ask for structured output with an explicit JSON schema.

**Visual reasoning: multi-step inference beyond what's visible.** Use a LLaVA-style model or Qwen-VL. Prompt with CoT, broken down into explicit steps.

**Chart and data QA: read and reason over charts.** Use Qwen-VL or a high-resolution LLaVA. Have the model extract the values first, then reason, and be honest about uncertainty.

Memorise these five. They're the lecture's own summary slide.

---

## Concept 1 — SAM recap and its loss function

Chapter 07 covered SAM. This lecture added one new slide, about **how SAM's masks are trained**.

### The problem

A segmentation mask is **mostly background**. With a plain pixel-by-pixel loss, the model can score well by just predicting "background" everywhere, and small objects get ignored.

### The fix: two losses, each for a different imbalance

SAM uses two losses, because there are two kinds of imbalance to deal with.

**Focal loss handles imbalance at the pixel level.** It down-weights the easy background pixels, so the hard pixels count for more.

**Dice loss handles overlap at the region level.** It measures how well the predicted shape overlaps the true shape, so a tiny object and a huge object matter equally.

SAM combines them with focal weighted much more heavily (**20 : 1**). A separate **mean-squared-error loss** trains the IoU head that scores the three candidate masks.

!!! example "Everyday analogy"
    Marking a colouring-in exercise. Focal loss is like a teacher who stops praising the easy big empty areas and looks closely at the tricky edges. Dice loss is like stepping back and asking "does the overall shape match?", whether the drawing is of an ant or an elephant.

### Two Q&A clarifications

- **MAE pre-training of SAM's encoder** predicts the *raw pixel values* of hidden patches, scored with a squared-error loss. Pixels are numbers, not choices from a vocabulary, so it isn't like predicting masked words.
- **The image encoder (not the decoder) is the expensive part.** It processes the whole 1024 × 1024 image once.

---

## Concept 2 — Image understanding and captioning

### What it is

**Captioning** means describing an image's content (objects, attributes, scene, context) in **free-form language**. It differs from **classification**, which picks a fixed label, and from **VQA**, which answers a specific question.

Uses: accessibility alt-text, product cataloguing, content moderation, social media captions and search indexing.

### Which model?

> **Need a sentence → a generative model (LLaVA/Qwen-style). Need only a tag or category → CLIP zero-shot.** It's cheaper and faster.

### How to prompt

- Use a **zero-shot, templated** prompt.
- **Constrain the scope and length**, for example: *"one sentence, under 125 characters, describing only what is visually present."*
- For cataloguing at scale, **keep the same template for every call**. **Consistency matters more than any single caption's creativity.**

!!! example "Worked example"
    **Prompt:** "Write a one-sentence caption describing only what is visually present in this image."
    **Output:** "A snow-capped mountain range under a clear sky, with the sun low on the horizon."

### What goes wrong

Captioning goes wrong in four typical ways, and each has its own fix.

**Hallucination: the caption mentions objects that aren't there.** Use "only what is visually present" prompts, and better (higher-resolution) models.

**Over-generic captions, such as "A photo of a scene."** **Add scope and detail constraints** to the prompt.

**Missing the key detail: the caption ignores the most important object.** Ask for the main subject explicitly.

**Inherited bias: the descriptions are stereotyped.** Review and curate the data.

**The lecture's model comparison** used a photo of a couple on a motorcycle carrying a satellite dish:

- Qwen-VL 2B was concise and noticed the dish.
- LLaVA-1.5 7B **hallucinated** ("the motorcycle is parked", "a handbag on the ground") and missed the dish.
- LLaVA-1.6 13B got the dish right and even read the licence plate. That's the benefit of higher resolution.

The lesson: **newer, bigger, higher-resolution models hallucinate less, but every model can.**

### Why it's hard, and how it's measured

Fine-grained differences, relations between objects, deciding what's important (which is subjective) and long-tail items (for example, local snack brands) all make captioning hard. Automatic metrics compare against human captions:

- **BLEU, ROUGE and METEOR** count word overlap.
- **SPICE** compares scene graphs.
- **CLIPScore** uses CLIP similarity.
- **CIDEr** is the common headline metric.

All of these correlate **poorly with human judgment**, which is why platforms like **CapArena** ask humans to pick the better of two captions.

### Datasets to recognise

**COCO Captions** (human-written gold standard, 5 captions per image), **Flickr30K**, **Conceptual Captions CC3M/CC12M** (noisy web alt-text), **Visual Genome** (dense regions and relationships) and **LAION-5B** (web-scale and noisy).

### How strong captioners are trained

1. Broad pre-training on noisy web pairs.
2. Curated supervised fine-tuning on human captions (COCO) for fluent style.
3. Instruction tuning (the LLaVA recipe) for conversational captioning.

**BLIP-style bootstrapping** trains a captioner and a filter together to clean up noisy web captions.

---

## Concept 3 — Visual Question Answering (VQA)

### What it is

**VQA** means answering a *specific, open-ended question* about an image. **The question decides what matters.** Question types include yes/no, multiple choice, counting and open-ended.

### Which model?

> **General whole-image question → LLaVA-style or Qwen-VL. Question about a specific region, or one that needs reading text in the image → Qwen-VL.**

### How to prompt

The right technique depends on the kind of question.

**For everyday questions, go zero-shot.** Just ask directly.

**For counting or comparison, use chain-of-thought.** For example: "First list every object and its colour, then count how many are red." Listing first makes every step auditable.

**For unusual domain vocabulary, go few-shot.** Show the model a few examples.

!!! example "Worked example"
    **Q:** "How many distinct peaks are visible in this image?"
    **A:** "Two: a smaller peak on the left and a taller, sunlit peak on the right."

### What goes wrong

VQA has three typical failures.

**Presupposition hallucination.** You ask "What colour is the cat?" when there's no cat, and the model answers "black". The fix is to **ground the prompt and explicitly allow "not present"**.

**Overconfidence, such as a confident wrong count.** Use CoT, and ask the model to verify its answer.

**Ambiguous questions, such as "Is this good?"** The question **lacks a defined criterion**, so rewrite it.

!!! example "Everyday analogy"
    Presupposition hallucination is like a witness asked "What colour was the getaway car?" who invents a colour instead of saying "I didn't see a car." The question itself planted the false assumption.

### Are VLMs even looking at the image?

Research shown in class compared answers on the **original image vs a fully blurred version**. Many answer words barely changed. Models often answer from **language habits (priors)** rather than the pixels. Attention heat maps also showed cases of right focus but wrong answer, and wrong focus but right answer.

From the Q&A: heat maps are read out of attention weights afterwards. **High attention on the right region doesn't guarantee the right answer.**

### Training and evaluation notes

- Start from a base VLM and fine-tune on mixed VQA data, with **answer-format supervision** (short answers where the benchmark expects them).
- Use **preference tuning (RLHF/DPO-style)** to reduce hallucination. Synthetic "wrong" examples are made by pairing an image with a random answer, a random question, or a **perturbed image** with objects blacked out. That last one "makes the V matter".
- Strict automatic VQA scoring **penalises answers that are correct but phrased differently**. GPT-4V scored poorly on such metrics despite strong human ratings.
- **Counting becomes unreliable beyond roughly 5–10 objects.**

### Datasets to recognise

**VQAv2** (general VQA), **GQA** (compositional questions built from scene graphs), **OK-VQA** (needs outside knowledge), **TextVQA** (needs reading text in the image) and **VizWiz** (real questions from blind and low-vision users).

---

## Concept 4 — Document intelligence

### What it is

**Extracting structured information** from invoices, forms, receipts and contracts. It combines **OCR (reading text)** with **layout understanding**. The output is **specific fields** for downstream systems, not prose.

### Which model?

> **Structured, layout-aware extraction → a VLM, specifically Qwen-VL** (native OCR + grounding). **Simple bulk text extraction → a dedicated OCR pipeline**, which is faster and cheaper but weaker at layout and relationships.

### Why documents need special care

Most image tasks work fine at low resolution (224 or 448 px). Documents don't, because the text is small and dense. Running external OCR and passing plain text to the model **loses the spatial layout**.

**Resolution is the biggest lever.** DocVQA scores jumped from about 65 (original Qwen-VL) to about 95 (Qwen2-VL) largely because of higher resolution and OCR-focused training data.

### How to prompt: give an explicit schema

- **Show the exact JSON shape.** Don't just describe it.
- **Name every field explicitly.**
- Ask for **one JSON object per document**.

!!! example "Worked example"
    **Prompt:** "Extract vendor, invoice number, date, and total. Respond only as JSON: {vendor, invoice_number, date, total}"
    **Output:** `{"vendor": "Acme Supplies Ltd.", "invoice_number": "INV-20394", "date": "2026-03-14", "total": "481.50"}`

!!! example "Everyday analogy"
    Handing a temp worker a blank form with labelled boxes ("Vendor: ___, Total: ___") instead of saying "write down the important stuff". The form gets you the same output from every document, every time.

### What goes wrong

Document extraction runs into four typical problems.

**Layout variability: every vendor's invoice looks different.** Use layout-aware VLMs and domain fine-tuning.

**Poor scans, with skew, blur or low resolution.** Pre-process first: deskew, crop and denoise, and use a higher resolution.

**Hallucinated fields.** If a field is missing, the model may invent a plausible value. Allow null or "not found" in the schema.

**High stakes, such as financial or legal documents.** **Route the output through human verification.**

Other open challenges: handwriting is weaker than print, multi-page reasoning, complex tables, numbers with no reliable confidence flag, and non-Latin scripts.

### Case studies (marked optional, but in the practice set)

- **DAVE:** a purpose-built document encoder (MAE-style pre-training on document images, then multi-task training with **learned model merging**) combined with a general encoder (SigLIP2). It improves document and web-agent benchmarks without hurting general VQA. Two details the practice set asks about:
    - **Raw pixels, not normalised patches.** Standard MAE normalises each patch before reconstructing it. Document and web pages are mostly flat white space, so patches have very **low variance**, and normalising them makes training unstable. DAVE reconstructs the **raw pixel values** instead.
    - **Learned merging, not plain averaging.** DAVE trains one encoder copy per LLM decoder (Qwen, Llama, Phi), then combines them with **learned merge coefficients** into one decoder-agnostic encoder. That beat simple weight averaging.
- **DocVLM:** encodes OCR text plus its layout positions and compresses them with **64 learnable queries** (the same idea as Qwen-VL's 256-query adapter) before feeding the frozen VLM.

### Datasets and metric to recognise

**DocVQA** (scored with **ANLS**, a similarity-based metric), **InfographicVQA**, **RVL-CDIP** (document-type classification) and **FUNSD** (noisy scanned forms, key–value extraction).

---

## Concept 5 — Visual reasoning

### What it is

**Multi-step inference beyond surface recognition:** spatial relations, counting, comparison, cause and effect. Unlike simple VQA, it requires **combining several observations**.

Uses: quality inspection, safety audits, interpreting diagrams and comparative analysis.

### Which model?

> **"The vision encoder gets you the pixels; the LLM does the reasoning."** Reasoning strength depends mostly on the **LLM backbone**, not the vision encoder. LLaVA-style and Qwen-VL both work; Qwen-VL has an edge when the reasoning refers to a specific region.

### How to prompt: chain-of-thought

Make the model **list the relevant observations before it answers**.

Compare two ways of asking the same question.

**A direct prompt: "Is it safe to park here?"** This carries a higher risk of a confident wrong yes or no.

**A CoT prompt: "First list every visible sign and marking, then decide if parking is allowed."** This brings the visible "No Parking" sign to the surface *before* the model answers.

!!! example "Everyday analogy"
    A maths teacher who insists you "show your working". The working lets you (and the teacher) spot exactly which step went wrong, instead of seeing only a wrong final number.

### What goes wrong

Reasoning fails in three typical ways.

**Compounding errors: one wrong step ruins the rest.** Break the problem into small steps.

**False confidence: steps that look rigorous but are wrong.** Check the steps, not just the answer.

**No self-check.** **Ask the model to verify its own steps.**

Visual reasoning has the **largest gap** between models and human experts. Benchmarks to recognise: **MMMU** (college-level), **MathVista**, **ScienceQA**, **NLVR2** and **Winoground** (where many models score near random).

Strong reasoning models are trained with a strong LLM backbone, step-by-step rationale data, reasoning-focused preference tuning or RL, and a curriculum that mixes VQA, maths, science and diagrams.

---

## Concept 6 — Chart & data QA

Charts need **both** image understanding (shapes and trends) **and** document understanding (axis labels, ticks, legends and titles via OCR). So use **Qwen-VL or a high-resolution LLaVA**.

The prompting strategy: **extract the values first, then reason, and be honest about uncertainty.** Models still struggle to read exact values off a chart.

---

## Composing models: three patterns from the Q&A

Three problems from the Q&A show how models can be combined.

**To detect a signature on a scanned document and cut it out precisely, chain a VLM into SAM.** A VLM (LLaVA or Qwen-VL) **locates** the signature and passes a box or point to SAM as a prompt. **SAM** then produces the pixel mask.

**To tell handwritten text from printed text, use Qwen-VL, not SAM.** Printed text has uniform letter shapes, even spacing and straight baselines, while handwriting varies.

**To check whether a design document's requirements are covered by test documents, split the job in two.** For **extraction**, use a document VLM with a named JSON schema per page. For **matching, don't use a VLM.** Split the text into atomic statements, embed them, retrieve and rerank, and have an LLM judge label each one "covered", "partially covered" or "not covered". Send low-confidence cases to human review.

The pattern: **use the VLM for what only a VLM can do (seeing), then hand off to cheaper text tools.**

Other practical answers from the Q&A:

- **Improving handwriting recognition:** higher-resolution scans, deskewing, cropping and denoising; fine-tuning on domain handwriting; confidence scores with manual review of low-confidence results.
- **Fine-tuning data for captioning:** a few thousand pairs to learn a style, 20–50K for a harder domain, and hundreds of millions if training from scratch.
- **Video:** the same ideas apply (patches, cross-attention) plus temporal memory, as in SAM 2. Control token count by **sampling frames** (for example 0.5–2 frames per second, or a fixed budget of 8–32 frames) and pooling tokens.

---

## Failure modes: quick memory list

Here are the typical failures for each task in one place.

**Captioning** suffers from hallucination and over-generic captions.

**VQA** suffers from presupposition hallucination, overconfidence and ambiguous questions.

**Documents** suffer from layout variability, poor scans and hallucinated fields, which is why they need human verification.

**Reasoning** suffers from compounding errors, false confidence and no self-check.

**Charts** suffer from misread exact values.

**A benchmark score tells you what a model did on that test, not on your data.** Always evaluate on your own documents.

---

## What the hands-on showed

!!! warning "Inferred"
    The Colab notebook was shared only as a link, and its contents are not in the course files. The planned lab was **"pick a task, choose a model, design the prompt"**, but the Q&A suggests only 10–20 minutes were available. The captioning comparison in the slides used Qwen-VL 2B, LLaVA-1.5 7B and LLaVA-1.6 13B, so the lab most likely used small models like these through Hugging Face. **This is unverified.**

A LoRA question from the previous day's lab also came up in the Q&A. A config with `target_modules=["q_proj","v_proj","v_proj","o_proj"]` had **"v_proj" listed twice. That's a typo**. Valid choices are `["q_proj","v_proj"]` or all four `["q_proj","k_proj","v_proj","o_proj"]`.

---

## Where students got confused (from the live Q&A)

These are real questions students asked during the session, with the answers they were given.

**"Does SAM use a vocabulary like CLIP?"** No. Its prompts are spatial (clicks, boxes, masks), and they can even be generated automatically as grid points.

**"Is a SAM prompt a question like 'what's in this picture'?"** No. It's a way of *pointing* at what to cut out. The output is a mask over the same image, not a zoom or a crop.

**"Is two-way attention something new?"** It's the same cross-attention idea used in translation, plus the reverse direction: image tokens also attend to prompt tokens.

**"What does MAE predict?"** The raw pixel values of the hidden patches, scored with a squared-error loss. It doesn't predict words from a vocabulary.

**"Which part of SAM is expensive?"** The image encoder, which runs ViT-H over the whole image. The decoder is tiny.

**"What's the difference between focal and Dice loss?"** Focal loss works at the pixel level and down-weights the easy background. Dice loss measures whole-shape overlap, so tiny and huge objects count equally.

**"How do I find a signature and segment it exactly?"** The VLM locates it, then SAM segments it from that box or point.

**"Can SAM tell handwriting from print?"** No. SAM doesn't recognise content, so use Qwen-VL.

**"Is chart reading image understanding or document understanding?"** Both. It needs high resolution too.

**"Do attention heat maps prove the model looked at the right thing?"** No. Focus is a by-product, and the right focus can still give a wrong answer.

---

## Traps to watch for in MCQs

- ❌ "Use a generative VLM to tag 10 million product photos into 20 categories." → **CLIP zero-shot** is cheaper and faster when only a tag is needed.
- ❌ "For cataloguing, vary the prompt to get more creative captions." → **Consistency beats creativity.** Use a fixed template.
- ❌ "Captions are too generic, so switch to a bigger model first." → First **add scope and detail constraints** to the prompt.
- ❌ "To stop presupposition hallucination, ask the question more forcefully." → **Ground the prompt and allow 'not present'.**
- ❌ "Visual reasoning ability comes mainly from the vision encoder." → It comes mainly from the **LLM backbone**.
- ❌ "For structured extraction, just describe the fields you want." → **Show the exact JSON schema** and name every field.
- ❌ "If the model can't find the total, it will leave it blank." → It may **hallucinate a plausible value**. Allow null and add human review.
- ❌ "Pure bulk text extraction always needs a VLM." → A **dedicated OCR pipeline** is faster and cheaper for that.
- ❌ "SAM can read and classify the text it segments." → SAM only produces masks. Pair it with a VLM.
- ❌ "A high benchmark score guarantees good results on our invoices." → Benchmarks **don't predict performance on your own data**.

---

## Recap

1. **Two decisions:** which model fits the task's shape, and which prompting technique fits the task.
2. **Captioning:** sentence → generative VLM, tag → CLIP. Scoped, length-limited, *fixed* templates. Watch for hallucination and generic captions.
3. **VQA:** region or OCR questions → Qwen-VL. CoT for counting and comparison. Fix presupposition hallucination by allowing "not present".
4. **Documents:** Qwen-VL + an **explicit JSON schema**. **Resolution** is the biggest lever. Expect hallucinated fields and keep a human in the loop.
5. **Reasoning and charts:** the **LLM does the reasoning**. Use CoT, list observations first, and ask the model to verify itself. For charts, extract values first, then reason.

---

## Practice questions

Pick an option to see instantly whether you're right, with an explanation for every option.

<div class="practice-quiz" data-quiz="ch08">
  <p><em>Loading practice questions…</em></p>
</div>

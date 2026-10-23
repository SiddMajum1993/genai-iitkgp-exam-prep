// Practice questions for Chapter 08 · Applications of VLMs.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch08"] = {
  "title": "Applications of VLMs",
  "questions": [
    {
      "id": "q1",
      "type": "Scenario",
      "stem": "An e-commerce company needs to sort 10 million product photos into 20 fixed categories (shoes, bags, watches…). No descriptions are needed. Which approach fits best?",
      "options": {
        "A": "CLIP zero-shot classification with the 20 category names as text prompts.",
        "B": "Qwen-VL with a JSON extraction schema.",
        "C": "SAM producing a mask for every product.",
        "D": "LLaVA-1.6 13B generating a detailed caption for every photo."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The lecture's rule: need a tag or category → contrastive (CLIP) zero-shot, which is cheaper and faster.",
        "B": "Qwen-VL shines at documents and grounding. For simple tagging it's needlessly heavy.",
        "C": "SAM outlines objects but doesn't name or categorise them.",
        "D": "This would work but is far slower and costlier. A generative model is overkill when only a tag is needed."
      },
      "takeaway": "Sentence needed → generative VLM. Tag only → CLIP."
    },
    {
      "id": "q2",
      "type": "Symptom → fix",
      "stem": "A captioning pipeline keeps producing captions like \"A photo of a scene.\" What is the first fix the lecture recommends?",
      "options": {
        "A": "Switch to a bigger model immediately.",
        "B": "Add scope and detail constraints to the prompt, e.g. \"one sentence describing only the main objects and their colours\".",
        "C": "Lower the image resolution so the model focuses.",
        "D": "Remove all instructions and let the model decide."
      },
      "correct": "B",
      "why": {
        "A": "A bigger model may help, but the lecture's quick-check answer for over-generic captions was to fix the prompt first.",
        "B": "Correct. Scoped, constrained prompts steer the model to the salient detail.",
        "C": "Lower resolution removes detail and usually makes captions worse.",
        "D": "Removing guidance makes generic captions more likely, not less."
      },
      "takeaway": "Over-generic captions → add scope and detail constraints to the prompt."
    },
    {
      "id": "q3",
      "type": "Symptom → fix",
      "stem": "A user asks a VQA assistant \"What colour is the cat?\" about a photo with no cat, and the model confidently answers \"Black.\" What is this failure and how do you reduce it?",
      "options": {
        "A": "Compounding error. Split the question into more steps.",
        "B": "Layout variability. Use a JSON schema.",
        "C": "Presupposition hallucination. Ground the prompt in what's visible and explicitly allow an answer like \"not present\".",
        "D": "Label noise. Retrain CLIP on cleaner captions."
      },
      "correct": "C",
      "why": {
        "A": "Compounding errors are a multi-step reasoning failure. This is a single false assumption.",
        "B": "Layout variability is a document-extraction problem.",
        "C": "Correct. The question assumes a cat exists. Letting the model say 'not present' removes the pressure to invent one.",
        "D": "Label noise is about training data. This is a prompting and inference-time failure."
      },
      "takeaway": "False assumption in the question → presupposition hallucination → allow 'not present'."
    },
    {
      "id": "q4",
      "type": "Best practice",
      "stem": "You're extracting vendor, invoice number, date and total from thousands of invoices with Qwen-VL. Which prompt follows the lecture's guidance?",
      "options": {
        "A": "\"Summarise this invoice in a short paragraph.\"",
        "B": "\"Extract vendor, invoice number, date and total. Respond only as JSON: {vendor, invoice_number, date, total}.\"",
        "C": "\"Think step by step about this invoice, then describe it.\"",
        "D": "\"Tell me everything important about this invoice.\""
      },
      "correct": "B",
      "why": {
        "A": "Prose can't be fed reliably into downstream systems.",
        "B": "Correct. Show the exact JSON shape, name every field, and ask for one object per document.",
        "C": "CoT suits reasoning tasks. Extraction needs a fixed schema.",
        "D": "Too vague. Output will vary from document to document and fields may be missed."
      },
      "takeaway": "Document extraction → explicit, named JSON schema."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "According to the lecture, what mainly determines how well a VLM handles multi-step visual reasoning, such as deciding whether parking is allowed from several signs?",
      "options": {
        "A": "The size of the vision encoder's patches.",
        "B": "The number of masks SAM produces.",
        "C": "The batch size used in CLIP pre-training.",
        "D": "The strength of the underlying LLM backbone."
      },
      "correct": "D",
      "why": {
        "A": "Patch size affects visual detail, but it isn't the main driver of reasoning.",
        "B": "SAM isn't involved in reasoning. It only segments.",
        "C": "CLIP's batch size affects its contrastive pre-training, not the reasoning of a generative VLM.",
        "D": "Correct. \"The vision encoder gets you the pixels; the LLM does the reasoning.\""
      },
      "takeaway": "Reasoning strength tracks the LLM backbone."
    },
    {
      "id": "q6",
      "type": "Scenario",
      "stem": "A bank wants to find signatures on scanned cheques and get an exact pixel outline of each one. Which design matches the Q&A answer?",
      "options": {
        "A": "Use a VLM (LLaVA/Qwen-VL) to locate the signature, then pass that box or point to SAM to get the pixel mask.",
        "B": "Use SAM alone with an automatic grid of points and keep every mask.",
        "C": "Use CLIP to output signature coordinates.",
        "D": "Use dedicated OCR to read the signature's text."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The VLM supplies the 'what and where' as a prompt, and SAM supplies the pixel-precise mask.",
        "B": "A grid would segment everything on the cheque. SAM alone doesn't know which mask is the signature.",
        "C": "CLIP produces similarity scores, not coordinates or masks.",
        "D": "Signatures aren't reliably readable text, and OCR doesn't give pixel outlines."
      },
      "takeaway": "Find it with a VLM, cut it out with SAM. Compose the models."
    },
    {
      "id": "q7",
      "type": "Risk",
      "stem": "An invoice is missing its due date, but your extraction schema includes a due_date field. What is the main risk, and what does the lecture recommend for high-stakes documents?",
      "options": {
        "A": "The model will copy the invoice date into every field. Use CoT.",
        "B": "The model will leave it blank, so no action is needed.",
        "C": "The model may hallucinate a plausible-looking date. Allow null values and route outputs through human verification.",
        "D": "The model will always crash. Remove the field."
      },
      "correct": "C",
      "why": {
        "A": "That's not the described failure, and CoT isn't the fix for extraction.",
        "B": "Assuming it will leave a blank is exactly the trap. Models often fill gaps with plausible values.",
        "C": "Correct. Hallucinated fields are a known document-intelligence pitfall, and financial or legal outputs need a human check.",
        "D": "Models don't crash on a missing field. The danger is that they quietly invent a value."
      },
      "takeaway": "Missing field → risk of hallucinated value → allow null and keep a human in the loop."
    },
    {
      "id": "q8",
      "type": "Why",
      "stem": "For the question \"How many red cars are in this parking lot?\", the lecture suggests a prompt like \"First list every car and its colour, then count the red ones.\" Why?",
      "options": {
        "A": "Chain-of-thought produces an auditable intermediate list, which makes counting more reliable and errors easier to spot.",
        "B": "It reduces the number of visual tokens.",
        "C": "It forces the model to use SAM internally.",
        "D": "It makes the model run faster."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Listing the evidence first makes the reasoning visible and checkable.",
        "B": "The image tokens are the same. CoT only changes the generated text.",
        "C": "Prompting doesn't add a segmentation model.",
        "D": "CoT produces more tokens, so it's usually slower, not faster."
      },
      "takeaway": "Counting and comparison → chain-of-thought: list the evidence, then answer."
    },
    {
      "id": "q9",
      "type": "Which is NOT",
      "stem": "Which of these is NOT good practice when captioning a large product catalogue with a VLM?",
      "options": {
        "A": "Limiting captions to one sentence describing only what is visible.",
        "B": "Varying the prompt wording for each image to get more creative captions.",
        "C": "Checking a sample of outputs for hallucinated objects.",
        "D": "Using the same prompt template for every image."
      },
      "correct": "B",
      "why": {
        "A": "This is good practice. Scope and length constraints reduce hallucination and rambling.",
        "B": "Correct, this is NOT good practice. The lecture said consistency matters more than any single caption's creativity.",
        "C": "This is good practice. Hallucination is the main known captioning failure.",
        "D": "This is good practice. A fixed template gives consistent output at scale."
      },
      "takeaway": "Captioning at scale: fixed template, consistency over creativity."
    },
    {
      "id": "q10",
      "type": "Concept",
      "stem": "SAM's mask training combines focal loss and Dice loss. What does Dice loss specifically add?",
      "options": {
        "A": "It down-weights easy background pixels.",
        "B": "It predicts raw pixel values for masked patches.",
        "C": "It trains the IoU head that scores the three masks.",
        "D": "It measures overall overlap between the predicted and true shapes, so small and large objects count equally."
      },
      "correct": "D",
      "why": {
        "A": "That's what focal loss does (pixel-level imbalance).",
        "B": "That describes MAE pre-training of the image encoder, not the mask loss.",
        "C": "The IoU head is trained with a separate mean-squared-error loss.",
        "D": "Correct. Dice is a region-level overlap measure that fixes object-size imbalance."
      },
      "takeaway": "Focal = pixel-level (easy background down-weighted). Dice = shape overlap (object size doesn't matter)."
    }
  ]
};

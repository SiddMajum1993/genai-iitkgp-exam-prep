// Practice questions for Chapter 07 · VLM architectures.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch07"] = {
  "title": "VLM architectures",
  "questions": [
    {
      "id": "q1",
      "type": "Concept",
      "stem": "During CLIP training, a batch contains many image–caption pairs. For a given image, where do its negative examples come from?",
      "options": {
        "A": "All the other captions in the same batch, which belong to other images.",
        "B": "Captions that were accidentally mismatched with images on the web.",
        "C": "Randomly generated nonsense sentences added to the batch.",
        "D": "Annotators hand-pick a few hard negative captions for every image."
      },
      "correct": "A",
      "why": {
        "A": "Correct. In the N × N grid, the diagonal is the true pairs and every other caption in the batch is a negative for that image.",
        "B": "Accidental mismatches in the data are label noise, not the designed negatives.",
        "C": "No synthetic text is added. The negatives are real captions of other images.",
        "D": "CLIP needs no manual negative mining. That's one of its big practical advantages."
      },
      "takeaway": "CLIP's negatives come free: every other pair in the batch. Bigger batches give more negatives."
    },
    {
      "id": "q2",
      "type": "Concept",
      "stem": "In LLaVA's first training stage (feature alignment), which component is trained?",
      "options": {
        "A": "Only the projector that maps image features into the LLM's word-embedding space.",
        "B": "Only the LLM.",
        "C": "The vision encoder and the projector.",
        "D": "All three: vision encoder, projector and LLM."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Stage 1 freezes both the CLIP encoder and the LLM and trains only the projector, so that image features land in the right place for the LLM.",
        "B": "The LLM is frozen in Stage 1. It is updated only in Stage 2.",
        "C": "LLaVA keeps the CLIP vision encoder frozen in both stages.",
        "D": "Nothing trains everything at once in LLaVA. The vision encoder stays frozen throughout."
      },
      "takeaway": "LLaVA: Stage 1 = projector only. Stage 2 = projector + LLM. The vision encoder is always frozen."
    },
    {
      "id": "q3",
      "type": "Scenario",
      "stem": "A team is building an assistant that must read text on product labels and answer questions like \"where on the box is the expiry date?\" by pointing to a region. Which architecture from the lecture fits best?",
      "options": {
        "A": "CLIP, because it is the fastest at image–text matching.",
        "B": "Original LLaVA, because it is the simplest chat model.",
        "C": "SAM, because it outputs precise pixel masks.",
        "D": "Qwen-VL, because it has OCR-focused high-resolution training and native grounding through box tokens."
      },
      "correct": "D",
      "why": {
        "A": "CLIP only scores similarity. It can't read text out or point to a region in an answer.",
        "B": "Original LLaVA has no built-in grounding and struggles with small text at its base resolution.",
        "C": "SAM can outline an object you point at, but it doesn't read text or understand the question.",
        "D": "Correct. Qwen-VL was designed for OCR and grounding: it writes coordinates inside <box> tags and trained at 448 px."
      },
      "takeaway": "Reading text or pointing at regions → Qwen-VL."
    },
    {
      "id": "q4",
      "type": "Compare",
      "stem": "How does Qwen-VL's connector differ from LLaVA's?",
      "options": {
        "A": "Qwen-VL uses a cross-attention adapter with 256 learnable queries, so it always produces 256 visual tokens. LLaVA's projector passes every patch through, so its token count grows with resolution.",
        "B": "Qwen-VL has no connector. Its ViT outputs words directly.",
        "C": "Both use the same projector. They differ only in the LLM.",
        "D": "Qwen-VL uses a bigger MLP projector; LLaVA uses cross-attention."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The 256 queries attend over all patch features (with 2D positions) and summarise them into a fixed-size set.",
        "B": "Every generative VLM needs a connector to bridge vision features into the LLM's space.",
        "C": "The connectors are fundamentally different, and that's the main architectural contrast.",
        "D": "This is reversed. LLaVA uses a linear or MLP projector; Qwen-VL uses cross-attention."
      },
      "takeaway": "Qwen-VL compresses to a fixed 256 tokens with cross-attention. LLaVA projects every patch."
    },
    {
      "id": "q5",
      "type": "Why",
      "stem": "SAM feels interactive: each new click returns a mask almost instantly. What design choice makes this possible?",
      "options": {
        "A": "SAM produces only one mask, which saves time.",
        "B": "SAM downsizes every image to a thumbnail before processing it.",
        "C": "SAM skips the image encoder entirely when given a point prompt.",
        "D": "The heavy image encoder runs once per image and its embedding is cached; each click only re-runs the lightweight prompt encoder and mask decoder."
      },
      "correct": "D",
      "why": {
        "A": "SAM returns three candidate masks per prompt, not one.",
        "B": "The speed comes from reusing the embedding, not from shrinking images.",
        "C": "The image embedding is always needed. It's just computed once and reused.",
        "D": "Correct. The encoder takes ~0.15 s once; the decoder takes ~50 ms per prompt, so repeated clicks are cheap."
      },
      "takeaway": "SAM = heavy encoder once, cheap decoder per prompt."
    },
    {
      "id": "q6",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT a prompt type that SAM accepts?",
      "options": {
        "A": "A rough mask to be refined.",
        "B": "A bounding box.",
        "C": "An audio clip describing the object.",
        "D": "A foreground or background point click."
      },
      "correct": "C",
      "why": {
        "A": "Rough masks are supported and are embedded with convolutions.",
        "B": "Boxes are a supported prompt for selecting one object precisely.",
        "C": "Correct, this is NOT a SAM prompt. Prompts are spatial (points, boxes, masks), with text only experimental.",
        "D": "Points are SAM's most common interactive prompt."
      },
      "takeaway": "SAM prompts: point, box, mask (text is experimental). Never audio."
    },
    {
      "id": "q7",
      "type": "Process",
      "stem": "To use CLIP for zero-shot classification into the classes 'cat', 'dog' and 'giraffe', what do you do with the class names?",
      "options": {
        "A": "Feed the class names into the image encoder alongside the photo.",
        "B": "Ask CLIP to generate the class name as text.",
        "C": "Turn each into a sentence like \"a photo of a cat\", embed it with the text encoder, and pick the class whose embedding is most similar to the image embedding.",
        "D": "Train a new classification head on labelled cat, dog and giraffe photos."
      },
      "correct": "C",
      "why": {
        "A": "Class names go through the text encoder, not the image encoder.",
        "B": "CLIP can't generate text. It only compares embeddings.",
        "C": "Correct. Classification becomes a nearest-neighbour search in the shared space, and prompt templates improve accuracy.",
        "D": "Training a head on labelled data is a linear probe, not zero-shot."
      },
      "takeaway": "CLIP zero-shot: class names → 'a photo of a {label}' → text embeddings → pick the closest one to the image."
    },
    {
      "id": "q8",
      "type": "Concept",
      "stem": "CLIP divides its similarity scores by a learned temperature τ before the softmax. What happens when τ is lowered?",
      "options": {
        "A": "The softmax becomes flatter, spreading probability evenly.",
        "B": "The softmax becomes sharper, so the model is more confident about the top match.",
        "C": "The embeddings become longer vectors.",
        "D": "The batch size effectively doubles."
      },
      "correct": "B",
      "why": {
        "A": "This is what a higher τ does.",
        "B": "Correct. A lower τ magnifies differences between scores, which gives a peakier distribution.",
        "C": "Embeddings are L2-normalised to length 1. τ only rescales the scores.",
        "D": "Temperature has nothing to do with batch size."
      },
      "takeaway": "Lower temperature = sharper; higher temperature = softer. CLIP learns τ."
    },
    {
      "id": "q9",
      "type": "Scenario",
      "stem": "A colleague says: \"Our LLM is already instruction-tuned on text, so we can skip LLaVA's Stage 2 and use it straight after Stage 1.\" What is the main problem with this plan?",
      "options": {
        "A": "Without Stage 2, the projector would be reset to random values.",
        "B": "Stage 1 already trains the LLM fully, so nothing is lost.",
        "C": "The LLM has never learned to interpret visual tokens or follow visual instructions, so Stage 2 is still needed to teach that.",
        "D": "Stage 2 is only needed to retrain the CLIP encoder."
      },
      "correct": "C",
      "why": {
        "A": "The projector isn't reset between stages. It carries over and keeps training in Stage 2.",
        "B": "The LLM is frozen in Stage 1. Only the projector is trained there.",
        "C": "Correct. This was asked in the Q&A. Text instruction tuning doesn't teach the model to reason over image tokens.",
        "D": "The CLIP encoder stays frozen in both stages."
      },
      "takeaway": "Stage 2 is required even for a text-instruction-tuned LLM, because it must learn to use visual tokens."
    },
    {
      "id": "q10",
      "type": "Scenario",
      "stem": "A photo-editing app wants users to type \"remove the red umbrella\" and get a pixel-perfect cut-out. SAM alone can't do this. Which design matches the lecture's recommendation?",
      "options": {
        "A": "Use CLIP to produce the mask directly.",
        "B": "Use LLaVA alone, since it can output masks.",
        "C": "Retrain SAM so that it outputs class names.",
        "D": "Use a grounding model or VLM to locate and name the umbrella (a box), then pass that box to SAM as a prompt to get the pixel mask."
      },
      "correct": "D",
      "why": {
        "A": "CLIP produces similarity scores, not masks.",
        "B": "Original LLaVA outputs text only. It needs another model to produce masks.",
        "C": "The lecture's approach is composition, not retraining SAM. SAM is a segmentation component that knows 'where', not 'what'.",
        "D": "Correct. This is the Grounded-SAM pattern: the grounding model gives the 'what' and a box, and SAM gives the precise 'where'."
      },
      "takeaway": "Text request → grounding model (box) → SAM (mask). Compose the models."
    }
  ]
};

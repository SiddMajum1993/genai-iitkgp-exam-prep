// Practice questions for Chapter 04 · LoRA & QLoRA.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch04"] = {
  "title": "LoRA & QLoRA",
  "questions": [
    {
      "id": "q1",
      "type": "Concept",
      "stem": "During LoRA fine-tuning of a pre-trained model, which statement correctly describes what gets updated?",
      "options": {
        "A": "The original weights are compressed into a low-rank form and then fully retrained.",
        "B": "All original weights are updated, but with a much smaller learning rate.",
        "C": "A small, carefully chosen subset of the original weights is updated; the rest are frozen.",
        "D": "The original weights stay frozen; only two new low-rank matrices A and B placed beside them are trained."
      },
      "correct": "D",
      "why": {
        "A": "LoRA doesn't compress W₀. The low-rank structure applies to the update ΔW, not to the original weights.",
        "B": "That is ordinary full fine-tuning with a low learning rate. Every weight still needs gradients and optimizer states.",
        "C": "Updating a chosen subset of existing weights is selective PEFT (e.g. BitFit). LoRA touches none of the original weights.",
        "D": "Correct. W₀ is frozen and the learned change is written as ΔW = B·A, so only A and B receive gradients."
      },
      "takeaway": "LoRA = frozen W₀ + trainable side path B·A. Zero original weights are changed."
    },
    {
      "id": "q2",
      "type": "Why",
      "stem": "B is initialised to zero in LoRA while A starts with small random values. What is the main reason?",
      "options": {
        "A": "To make the adapter file smaller when it is saved to disk.",
        "B": "So that the product B·A is zero at the start and the model initially behaves exactly like the pre-trained model.",
        "C": "To force the model to forget its pre-training before learning the new task.",
        "D": "Because zero-initialised matrices train faster on GPUs."
      },
      "correct": "B",
      "why": {
        "A": "The saved file size depends on the shape of A and B, not on their starting values.",
        "B": "Correct. With B = 0, ΔW = 0, so training starts from the known-good pre-trained behaviour and moves away gently.",
        "C": "The opposite. The zero start exists precisely to preserve pre-trained behaviour at step 0.",
        "D": "Initial values don't change GPU speed. (Also, if both A and B were zero, gradients would get stuck.)"
      },
      "takeaway": "B = 0 means the model starts as 'pre-trained model + nothing'. It's the same idea as near-identity adapter initialisation."
    },
    {
      "id": "q3",
      "type": "Scenario",
      "stem": "A team fine-tuned a model for a single customer-facing task and needs the lowest possible inference latency. They are choosing between Houlsby-style adapters and LoRA. What is the strongest argument for LoRA here?",
      "options": {
        "A": "LoRA removes layers from the model, making it faster than the original.",
        "B": "LoRA trains without needing any GPU.",
        "C": "LoRA's update can be merged into W₀ after training, so the deployed model has the same structure and speed as the original.",
        "D": "LoRA always reaches higher accuracy than adapters."
      },
      "correct": "C",
      "why": {
        "A": "A merged LoRA model is exactly as fast as the original. It isn't faster, because no layers are removed.",
        "B": "LoRA still trains on a GPU. It just needs much less memory.",
        "C": "Correct. W_merged = W₀ + (α/r)·B·A is computed once, offline. Adapters stay in the network and add sequential computation on every call.",
        "D": "The course doesn't claim LoRA always wins on accuracy. The two are often comparable."
      },
      "takeaway": "Merge ability = zero added inference latency. This is the key advantage of re-parameterisation over additive methods."
    },
    {
      "id": "q4",
      "type": "Scenario",
      "stem": "A SaaS company serves 200 business clients, each needing its own fine-tuned behaviour from the same 13B base model. Which deployment design best matches what the course described?",
      "options": {
        "A": "Keep one base model in GPU memory and hot-swap the right client's small LoRA adapter per request, chosen by a router using the client ID.",
        "B": "Train one shared adapter for all clients and rank clients by importance.",
        "C": "Merge all 200 LoRA adapters into a single set of weights.",
        "D": "Fully fine-tune and host 200 separate copies of the 13B model."
      },
      "correct": "A",
      "why": {
        "A": "Correct. One resident base model plus a ~50 MB adapter per client (~10 GB total), selected per request.",
        "B": "Adapters aren't 'ranked'. Each client has its own adapter, selected by client ID.",
        "C": "Merging all adapters together would blend every client's behaviour into one model and lose per-client customisation.",
        "D": "This is the naive approach the lecture rejected: about 5.2 TB of storage and hundreds of warm GPUs."
      },
      "takeaway": "One base + many small swappable adapters is the multi-tenant LoRA pattern. Merge for one task; keep separate for many."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "Why does LoRA reduce GPU memory during training so dramatically compared with full fine-tuning?",
      "options": {
        "A": "Because the LoRA adapter file is only ~50 MB on disk.",
        "B": "Because LoRA deletes the activations after every layer.",
        "C": "Because the frozen base weights need no gradients and no optimizer (Adam) states. Those exist only for the tiny A and B matrices.",
        "D": "Because the base model is stored in 4 bits."
      },
      "correct": "C",
      "why": {
        "A": "Small file size is a nice side effect for storage, but it isn't why GPU training memory drops.",
        "B": "Discarding and recomputing activations is gradient checkpointing, a separate technique.",
        "C": "Correct. In full fine-tuning, gradients plus Adam's two states cost more memory than the weights themselves. LoRA skips all of that for W₀.",
        "D": "4-bit storage of the base is QLoRA's addition, not plain LoRA's."
      },
      "takeaway": "LoRA saves memory by not training W₀: no gradients and no optimizer states for the frozen base."
    },
    {
      "id": "q6",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT one of the memory-saving ingredients that QLoRA adds on top of LoRA?",
      "options": {
        "A": "Paged optimizer that moves optimizer states between CPU RAM and GPU.",
        "B": "Quantizing the trainable LoRA matrices A and B to 4 bits.",
        "C": "Double quantization of the block-wise scaling constants.",
        "D": "Storing the frozen base weights in 4-bit NormalFloat (NF4)."
      },
      "correct": "B",
      "why": {
        "A": "This is a QLoRA ingredient: it absorbs memory spikes instead of crashing with OOM.",
        "B": "Correct, this is NOT done. A and B stay in higher precision (BF16). Only the frozen W₀ is quantized.",
        "C": "This is a QLoRA ingredient: it compresses the constants created by block-wise quantization.",
        "D": "This is a QLoRA ingredient: the biggest memory item, the base model, shrinks about 4×."
      },
      "takeaway": "QLoRA quantizes only the frozen base. The trainable A and B stay in BF16."
    },
    {
      "id": "q7",
      "type": "Symptom → fix",
      "stem": "A QLoRA run on a single GPU usually trains fine but occasionally crashes with an out-of-memory error, always on batches that contain an unusually long sequence. Which technique is designed for exactly this problem?",
      "options": {
        "A": "Merging the adapter into W₀.",
        "B": "Double quantization.",
        "C": "Paged optimizer.",
        "D": "Increasing the LoRA rank r."
      },
      "correct": "C",
      "why": {
        "A": "Merging is a post-training step for deployment. It does nothing for training memory.",
        "B": "Double quantization gives a steady, small saving on constants. It doesn't absorb sudden spikes.",
        "C": "Correct. The paged optimizer spills optimizer states to CPU RAM when GPU memory spikes, instead of crashing.",
        "D": "A higher rank adds trainable parameters and uses more memory, which makes things worse."
      },
      "takeaway": "Sudden OOM spikes → paged optimizer. Steady activation memory → gradient checkpointing."
    },
    {
      "id": "q8",
      "type": "Compare",
      "stem": "Block-wise quantization and NF4 are both used in QLoRA. A student says they are redundant. What is the best response?",
      "options": {
        "A": "Block-wise quantization is for the optimizer states, NF4 is for the gradients.",
        "B": "Block-wise quantization limits the damage from outliers by giving each small block its own scale; NF4 places the 16 levels to match the bell-shaped weight distribution. They solve different problems.",
        "C": "NF4 is used for the forward pass and block-wise quantization only for the backward pass.",
        "D": "They are redundant. QLoRA uses one or the other depending on the model."
      },
      "correct": "B",
      "why": {
        "A": "Both apply to the frozen base weights, not to optimizer states or gradients.",
        "B": "Correct. Block-wise handles scale (an outlier only spoils its own block); NF4 handles shape (more levels near zero, where most weights are).",
        "C": "Both describe how the frozen W₀ is stored. They aren't split between forward and backward passes.",
        "D": "QLoRA uses both together. They address different weaknesses."
      },
      "takeaway": "Block-wise = per-block scale (fixes outliers). NF4 = level placement (fits the bell curve). Complementary."
    },
    {
      "id": "q9",
      "type": "Code reading",
      "stem": "Look at this configuration. Which statement about it is correct?",
      "code": "bnb_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type=\"nf4\",\n    bnb_4bit_use_double_quant=True,\n    bnb_4bit_compute_dtype=torch.bfloat16,\n)\nlora_config = LoraConfig(\n    r=16, lora_alpha=32,\n    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n)",
      "options": {
        "A": "This sets up QLoRA: a 4-bit NF4 base with double quantization, computation in BF16, and LoRA adapters of rank 16 on the four attention projections.",
        "B": "lora_alpha=32 means the adapter has 32 trainable layers.",
        "C": "The LoRA matrices themselves will be stored in 4-bit NF4.",
        "D": "This is plain LoRA. The base model is kept in 16-bit."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Every line matches a QLoRA ingredient: NF4, double quantization, BF16 compute, and LoRA on Q/K/V/O.",
        "B": "α is the scaling strength (used as α/r), not a count of layers.",
        "C": "Quantization applies to the frozen base. The LoRA A and B matrices are trained in higher precision.",
        "D": "load_in_4bit=True with nf4 means the base is quantized. That is QLoRA, not plain LoRA."
      },
      "takeaway": "4-bit + nf4 + double_quant + BF16 compute + LoraConfig = QLoRA in code."
    },
    {
      "id": "q10",
      "type": "Concept",
      "stem": "What does the finding about 'intrinsic dimension' tell us, and why does it support LoRA?",
      "options": {
        "A": "Higher-dimensional models always need higher LoRA ranks.",
        "B": "Big models can be permanently shrunk to a few hundred parameters without losing ability.",
        "C": "Fine-tuning works best when every parameter is changed by the same amount.",
        "D": "Adapting a pre-trained model to a new task needs only a small number of effective directions of change, so a low-rank update can capture most of what full fine-tuning would."
      },
      "correct": "D",
      "why": {
        "A": "The course's advice is to pick the smallest r that validates well. Rank isn't tied directly to model size.",
        "B": "The model keeps all its parameters. Intrinsic dimension is about the size of the needed change, not the model.",
        "C": "Nothing in the theory says this. It's about how few directions the change needs.",
        "D": "Correct. For example, BERT has 110M parameters but needs only ~200 directions to solve MRPC. A simple change fits a low-rank ΔW."
      },
      "takeaway": "Intrinsic dimension: the task-specific change is simple, which is why a low-rank B·A is enough."
    }
  ]
};

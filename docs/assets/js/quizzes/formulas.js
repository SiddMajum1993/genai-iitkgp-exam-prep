// Practice questions for the Exam 2 toolkit page · Numbers & formulas.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["formulas"] = {
  "title": "Numbers & formulas",
  "questions": [
    {
      "id": "q1",
      "type": "Calculation",
      "stem": "Using the course's rule of 12 bytes per parameter (FP16 weight + FP16 gradient + Adam's two FP32 states), roughly how much memory does full fine-tuning of a 13B-parameter model need, before activations?",
      "options": {
        "A": "26 GB",
        "B": "156 GB",
        "C": "52 GB",
        "D": "208 GB"
      },
      "correct": "B",
      "why": {
        "A": "26 GB is 13B × 2 bytes, which is only the FP16 weights. It's what you need to store or serve the model, not to train it.",
        "B": "Correct. 13 billion × 12 bytes = 156 GB.",
        "C": "52 GB is 13B × 4 bytes, the weights in FP32. It leaves out the gradients and Adam's states.",
        "D": "208 GB is 13B × 16 bytes, the count that adds an FP32 master copy. The question fixes the rule at 12 bytes."
      },
      "takeaway": "Full fine-tuning ≈ 12 bytes per parameter (16 with an FP32 master copy). Adam's states are the biggest part."
    },
    {
      "id": "q2",
      "type": "Calculation",
      "stem": "LoRA with rank r = 16 is applied to one 4096 × 4096 weight matrix. How many trainable parameters does LoRA add for this matrix?",
      "options": {
        "A": "65,536",
        "B": "16,777,216",
        "C": "4,096",
        "D": "131,072"
      },
      "correct": "D",
      "why": {
        "A": "65,536 is the answer for r = 8. Doubling the rank doubles the count.",
        "B": "16,777,216 is 4096 × 4096, the full matrix. That's what full fine-tuning would train.",
        "C": "4,096 is just one side of the matrix. LoRA trains B (4096 × 16) and A (16 × 4096).",
        "D": "Correct. r × (d + k) = 16 × (4096 + 4096) = 16 × 8192 = 131,072, about 0.8% of the full matrix."
      },
      "takeaway": "LoRA parameters = r × (d + k). The count grows linearly with r."
    },
    {
      "id": "q3",
      "type": "Calculation",
      "stem": "A sequential adapter has hidden size d = 1024 and bottleneck size m = 32. Using the course formula 2md + m + d, how many parameters does one adapter add?",
      "options": {
        "A": "66,592",
        "B": "65,536",
        "C": "32,768",
        "D": "1,056"
      },
      "correct": "A",
      "why": {
        "A": "Correct. 2 × 32 × 1024 = 65,536 weights, plus 32 + 1024 = 1,056 biases, gives 66,592.",
        "B": "65,536 counts only the two weight matrices (2md). It leaves out the biases (m + d).",
        "C": "32,768 is md, only one of the two projections (down or up).",
        "D": "1,056 is m + d, only the biases."
      },
      "takeaway": "Adapter = down-projection (d×m + m) + up-projection (m×d + d) = 2md + m + d."
    },
    {
      "id": "q4",
      "type": "Calculation",
      "stem": "A training run uses a per-GPU micro-batch of 4, gradient accumulation of 8 steps, and 2 GPUs. What is the effective batch size?",
      "options": {
        "A": "14",
        "B": "32",
        "C": "64",
        "D": "8"
      },
      "correct": "C",
      "why": {
        "A": "14 adds the three numbers. They multiply.",
        "B": "32 is micro-batch × accumulation on one GPU. It forgets the second GPU.",
        "C": "Correct. 4 × 8 × 2 = 64.",
        "D": "8 is the micro-batch summed across the two GPUs. It forgets gradient accumulation."
      },
      "takeaway": "Effective batch = micro-batch × accumulation steps × number of GPUs."
    },
    {
      "id": "q5",
      "type": "Calculation",
      "stem": "A LoRA run works well with batch size 16 and learning rate 1e-4. You raise the effective batch to 64. Following the linear scaling rule, what learning rate should you start from?",
      "options": {
        "A": "4e-4",
        "B": "2.5e-5",
        "C": "1e-4",
        "D": "2e-4"
      },
      "correct": "A",
      "why": {
        "A": "Correct. The batch grew 64 ÷ 16 = 4 times, so the learning rate grows 4 times: 1e-4 × 4 = 4e-4.",
        "B": "2.5e-5 divides by 4. The rule scales the learning rate up with the batch, not down.",
        "C": "Keeping the same learning rate ignores the rule. A bigger batch gives a less noisy gradient, so it can take bigger steps.",
        "D": "2e-4 would be right if the batch had only doubled."
      },
      "takeaway": "Linear scaling: η_new ≈ η_old × B_new / B_old."
    },
    {
      "id": "q6",
      "type": "Calculation",
      "stem": "You quantize X = [1.0, −4.0, 2.0] to 8 bits with absmax quantization: c = 127 / max|X| and q = round(c · x). What integer is stored for the value 1.0?",
      "options": {
        "A": "127",
        "B": "32",
        "C": "31",
        "D": "25"
      },
      "correct": "B",
      "why": {
        "A": "127 is stored only for the value with the largest magnitude, here −4.0 (stored as −127).",
        "B": "Correct. max|X| = 4.0, so c = 127 ÷ 4 = 31.75, and round(31.75 × 1.0) = 32.",
        "C": "31 truncates 31.75 instead of rounding it.",
        "D": "25 would come from using 100 instead of 127 as the int8 limit."
      },
      "takeaway": "Absmax: c = 127 / max|X|, store round(c·x) plus c. The largest value maps to ±127."
    },
    {
      "id": "q7",
      "type": "Calculation",
      "stem": "You fine-tune on 20,000 examples with an effective batch size of 32 for 2 epochs, with warmup_ratio = 0.04. How many warm-up steps is that?",
      "options": {
        "A": "800",
        "B": "25",
        "C": "1,250",
        "D": "50"
      },
      "correct": "D",
      "why": {
        "A": "800 applies the ratio to the number of examples (20,000 × 0.04), not to optimizer steps.",
        "B": "25 uses only one epoch (625 × 0.04). The total covers both epochs.",
        "C": "1,250 is the total number of steps, not the warm-up part.",
        "D": "Correct. 20,000 ÷ 32 = 625 steps per epoch, × 2 epochs = 1,250 total steps, × 0.04 = 50 warm-up steps."
      },
      "takeaway": "Warm-up steps = warmup_ratio × total optimizer steps, and total steps = (examples ÷ effective batch) × epochs."
    },
    {
      "id": "q8",
      "type": "Calculation",
      "stem": "A ViT splits a 448 × 448 image into 14 × 14 patches. How many patches (not counting any [CLS] token) does it produce?",
      "options": {
        "A": "32",
        "B": "256",
        "C": "1,024",
        "D": "196"
      },
      "correct": "C",
      "why": {
        "A": "32 is 448 ÷ 14, the number of patches along one side only.",
        "B": "256 is the count for a 224 × 224 image with 14 × 14 patches.",
        "C": "Correct. (448 ÷ 14)² = 32² = 1,024.",
        "D": "196 is the count for 224 × 224 with 16 × 16 patches."
      },
      "takeaway": "Patches = (H ÷ P) × (W ÷ P). Doubling the image side quadruples the patch count."
    },
    {
      "id": "q9",
      "type": "Calculation",
      "stem": "In block-wise quantization, each block of 256 weights stores one 32-bit quantization constant. How many extra bits per parameter does this add?",
      "options": {
        "A": "0.5",
        "B": "0.125",
        "C": "8",
        "D": "0.25"
      },
      "correct": "B",
      "why": {
        "A": "0.5 is the overhead for blocks of 64.",
        "B": "Correct. 32 bits ÷ 256 weights = 0.125 bits per parameter.",
        "C": "8 is 256 ÷ 32, which divides the wrong way round.",
        "D": "0.25 is the overhead for blocks of 128."
      },
      "takeaway": "Overhead per parameter = bits per constant ÷ block size. Bigger blocks mean less overhead but less precision."
    },
    {
      "id": "q10",
      "type": "Calculation",
      "stem": "Gradient clipping uses max_grad_norm = 1.0. In one step the gradient vector has norm 5. What happens to the gradient?",
      "options": {
        "A": "It is multiplied by 0.2, so its norm becomes 1 and its direction stays the same.",
        "B": "The step is skipped because the norm is above the limit.",
        "C": "Every component larger than 1.0 is set to 1.0.",
        "D": "It is multiplied by 5 to make its norm 25."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The scale factor is min(1, 1.0 ÷ 5) = 0.2, so the norm becomes 5 × 0.2 = 1. Every component shrinks by the same factor, so the direction is unchanged.",
        "B": "Clipping never skips the update. It shrinks it.",
        "C": "That would be clipping each value separately, which changes the direction. Norm clipping rescales the whole vector.",
        "D": "Clipping only ever shrinks a gradient, never enlarges it."
      },
      "takeaway": "g ← g × min(1, c / ‖g‖): same direction, norm capped at c."
    },
    {
      "id": "q11",
      "type": "Calculation",
      "stem": "A reward model is trained with loss L = −log σ(r_chosen − r_rejected). Early in training it gives the chosen and rejected answers exactly the same score. What is the loss for this pair?",
      "options": {
        "A": "0",
        "B": "1",
        "C": "0.13",
        "D": "About 0.69 (that is, −log 0.5)"
      },
      "correct": "D",
      "why": {
        "A": "A loss of 0 would need σ(gap) = 1, which happens only when the chosen answer scores far higher.",
        "B": "1 doesn't come from the formula. σ(0) is 0.5, not 1/e.",
        "C": "0.13 is the loss when the chosen answer scores about 2 higher.",
        "D": "Correct. The gap is 0, σ(0) = 0.5, and −log 0.5 ≈ 0.69."
      },
      "takeaway": "Equal scores give a loss of 0.69. A bigger correct gap shrinks the loss, and a wrong-way gap makes it large."
    },
    {
      "id": "q12",
      "type": "Calculation",
      "stem": "In RLHF the policy's reward is r_total = r_φ − β · KL(π_θ ‖ π_ref). For one response the reward model gives 6, the KL divergence from the reference is 2, and β = 0.5. What is r_total?",
      "options": {
        "A": "7",
        "B": "4",
        "C": "5",
        "D": "3"
      },
      "correct": "C",
      "why": {
        "A": "7 adds the penalty. The KL term is subtracted.",
        "B": "4 subtracts the KL value without multiplying by β.",
        "C": "Correct. 6 − 0.5 × 2 = 6 − 1 = 5.",
        "D": "3 multiplies the reward by β instead of the KL term."
      },
      "takeaway": "A larger β or a larger drift from the reference lowers the reward, which keeps the policy close to the reference."
    }
  ]
};

# 9-Day Study Schedule — upGrad × IIT KGP GenAI Exam

**Today:** Wed 9 July 2026 · **Exam:** Fri 18 July 2026
**Target:** Pass with buffer (~17/24 = 70%)
**Total budget:** ~25 hours across 9 days

You have three tiers of material to work through:

| Tier | Folder | Purpose | When to use |
|---|---|---|---|
| 1. Learn | `learn_guides/` | Teach the concept from scratch. Plain English, analogies, worked examples. 20–35 min per lecture. | Days 1–7 (primary reading) |
| 2. Revise | `study_notes/` | Compressed formulas, tables, flashcards. | Day 8 (after you've learned it) |
| 3. Skim | `EXAM_CHEATSHEET.md` | One-page mental refresh. | Day 9 (day before exam) |

If you only ever read one thing per lecture: read the **learn_guide**. The rest is compression.

Every day has a **must-do** minimum and an **if-time-permits** stretch. If a work day blows up, do the must-do only — you'll still pass.

---

## Day 1 — Wed 9 July (today, evening) · 45 min

**Focus:** Orient yourself. Understand what you're facing. No new content yet.

**Must-do:**
- [ ] Read `EXAM_CHEATSHEET.md` end-to-end once. Don't try to memorise — just see the shape of the exam and what topics exist.
- [ ] Open `Sample_Paper/all-sample-questions-formatted-v2.pdf`. Read Q1 through Q6 (Section 1) with the answers COVERED. Note which felt easy vs unfamiliar. This gives you a baseline.
- [ ] No new lecture reading tonight. Sleep on it.

---

## Day 2 — Thu 10 July · 2 hours

**Focus:** ML foundations (Lecture 1) + neural network deep dive (Lecture 2)

**Must-do (~90 min):**
- [ ] Read `learn_guides/Lecture_01_learn.md` fully. Do the 5 "Check yourself" questions at the end BEFORE looking at answers.
- [ ] Read `learn_guides/Lecture_02_learn.md` fully. Same drill on the questions.

**If time (~30 min):**
- [ ] Do a confusion matrix drill on paper: given TP=80, FP=20, FN=10, TN=890, compute Precision, Recall, F1, Accuracy. Answers: 0.80, 0.89, 0.84, 0.97.
- [ ] Do sample paper Q1-Q5. Score yourself.

---

## Day 3 — Fri 11 July · 2 hours

**Focus:** Transformers (Lecture 3) + LLM decoding & APIs (Lecture 4)

**Must-do (~90 min):**
- [ ] Read `learn_guides/Lecture_03_learn.md` — the Q/K/V attention explanation is the most important concept in the entire course. If it doesn't click on first pass, reread just Concept 3 (self-attention) again.
- [ ] Read `learn_guides/Lecture_04_learn.md` — decoding strategies and OpenAI API. Pay special attention to the top-k ∩ top-p worked example.

**If time (~30 min):**
- [ ] Recreate the top-k ∩ top-p problem on paper WITHOUT the guide. Given probs A=0.13, B=0.04, C=0.22, D=0.16, E=0.01, F=0.18, G=0.07, H=0.10, I=0.06, J=0.03 with k=2, p=0.95, what's the intersection? (Answer: {C, F}, top-k binding.)
- [ ] Do sample paper Q6-Q10. Score yourself.

---

## Day 4 — Sat 12 July (WEEKEND — long session) · 3 hours

**Focus:** Structured outputs, LLM evaluation, AI safety (Lectures 5 & 6)

**Must-do (~2 hrs):**
- [ ] Read `learn_guides/Lecture_05_learn.md`. Build a small table on paper: which metric (BLEU/ROUGE/METEOR/BERTScore) for which task (translation, summarisation, paraphrase, morphology). Test yourself.
- [ ] Read `learn_guides/Lecture_06_learn.md`. Build another table: which attack type violates which CIA property. Memorise these.

**If time (~1 hr):**
- [ ] Do sample paper Q11-Q12. Score yourself.
- [ ] Read the "Common misunderstandings" section of each concept in both learn-guides — these are the exam traps.

---

## Day 5 — Sun 13 July (WEEKEND — long session) · 3 hours

**Focus:** Advanced prompting techniques + prompt security (Lectures 7 & 8)

**Must-do (~2 hrs):**
- [ ] Read `learn_guides/Lecture_07_learn.md`. This is the highest-leverage lecture for pattern-recognition MCQs. For each prompting technique, cover the "how to distinguish it" note and try to name it from just the prompt example.
- [ ] Read `learn_guides/Lecture_08_learn.md`. Pay attention to the APE log-probability arithmetic (closest to zero wins).

**If time (~1 hr):**
- [ ] Levenshtein distance drill on paper: HORSE→ROSE (2), KITTEN→SITTING (3), FLAW→LAWN (2). Note that HEART→EARTH is 2 via delete-H + insert-H, NOT 4 by naive substitution.
- [ ] Do sample paper Q13-Q16. Score yourself.

---

## Day 6 — Mon 14 July · 2 hours

**Focus:** RAG fundamentals + dense retrieval (Lectures 9 & 10)

**Must-do (~90 min):**
- [ ] Read `learn_guides/Lecture_09_learn.md`. Focus on the "naive vs advanced vs agentic RAG" distinction — this exact distinction is asked in the exam.
- [ ] Read `learn_guides/Lecture_10_learn.md`. Understand bi-encoder vs cross-encoder and the two-stage retrieve→rerank pattern.

**If time (~30 min):**
- [ ] Write a paragraph in your own words: "The core mechanism of RAG is..." — no looking. Then check against Lecture 9's answer.
- [ ] Do sample paper Q17-Q18. Score yourself.

---

## Day 7 — Tue 15 July · 2.5 hours · **HIGHEST-YIELD DAY**

**Focus:** Vector indexing + PQ math + agentic RAG (Lectures 11 & 12). 6 of the 24 sample paper questions come from here.

**Must-do (~2 hrs):**
- [ ] Read `learn_guides/Lecture_11_learn.md`. Work through the PQ math TWICE — once with the given sample-paper numbers (N=10M, D=1024, m=16, k=256), once with fresh numbers (N=5M, D=768, m=12, k=256).
- [ ] Read `learn_guides/Lecture_12_learn.md`. Understand the Observe-Think-Act loop and what makes agentic RAG different from linear RAG.

**Must-do drill (~30 min):**
- [ ] Redo the PQ math THREE TIMES from memory on paper. For N=10M, D=1024, m=16, k=256:
  - Raw storage = D × 4 × N = 40.96 GB
  - PQ storage = m × 1 × N = 160 MB
  - Lookup table = m × k = 4,096 entries
  - Distance per DB vector = m lookups + (m−1) additions = 16 + 15
- [ ] Do sample paper Q19-Q24. All 6 should feel achievable now.

---

## Day 8 — Wed 16 July · 2 hours

**Focus:** Consolidation. Move from `learn_guides/` to `study_notes/`.

**Must-do (~90 min):**
- [ ] Take a full timed mock exam. (If Mock_Exam_1.md exists in `practice/`, use that. Otherwise redo the sample paper cold.) 90-minute timer, 24 questions.
- [ ] Grade honestly. Identify weakest section (1, 2, 3, or 4).
- [ ] For every wrong answer: go back to the relevant `learn_guides/Lecture_XX_learn.md` and reread just that concept block.

**If time (~30 min):**
- [ ] Skim `study_notes/` for your weakest section only. Use these as compressed reminders, not new material.

---

## Day 9 — Thu 17 July · 60–90 min

**Focus:** Skim only. No new material. Rest.

**Must-do (~60 min):**
- [ ] Read `EXAM_CHEATSHEET.md` end-to-end TWICE. This is the single most valuable hour of the week.
- [ ] Reread only the "Quick reference" section of each `learn_guides/Lecture_XX_learn.md` (last section of each file). Skip everything else.
- [ ] Reread the sample paper QUESTIONS one more time (cover the answers — recall them from memory).

**Do NOT:**
- Take another mock exam (fatigue > gain).
- Start any new topic.
- Stay up late.

**Sleep by 11 PM.**

---

## Exam morning — Fri 18 July

**Before the exam:**
- Eat breakfast.
- Log in 20 min early.
- Have scratch paper + a working pen ready (for PQ math, Levenshtein, top-k∩top-p arithmetic).
- Test your webcam and mic before the proctor connects.

**During the exam:**
- Read the stem twice, options once.
- Do the arithmetic questions on paper — don't trust intuition.
- Flag hard questions; come back later.
- **No negative marking → answer ALL 24, even the ones you're guessing on.**
- Budget: 3:30 per question. Save 15 min for review.

**If you have to guess:**
- Eliminate the two obviously-wrong options.
- Between the last two: the option that names a specific mechanism (e.g., "injects RBAC metadata filter at the retrieval gateway") is usually the correct one over vague answers (e.g., "the system prompt tells the LLM to refuse").

---

## Escape hatches (if life happens)

**If you lose 1 weekday** (Days 2/3/6): Merge that day's lectures into the next day. Skip practice questions until Day 8's mock.

**If you lose a weekend day** (Days 4/5): Move that day's content to Days 6+7 evenings. Cut Day 8's mock in half if needed.

**If you have <4 days left:**
1. Read `EXAM_CHEATSHEET.md` (1 hour)
2. Read `learn_guides/Lecture_11_learn.md` and `Lecture_12_learn.md` in full (2 hours) — these alone cover 6 questions
3. Skim `learn_guides/` for the other 10 lectures — read only "What this lecture is about," "Common misunderstandings," and "Quick reference" sections of each
4. Redo the sample paper the day before

**If you have <1 day left:**
- Read `EXAM_CHEATSHEET.md` twice.
- Read the sample paper questions AND answers (both).
- Aim for 12/24 (pass). Don't overreach.

---

## Progress tracker

| Day | Date | Done? | Learn-guide read | Sample paper Qs done |
|---|---|---|---|---|
| 1 | Wed 9 Jul | ☐ | Cheat sheet only | — |
| 2 | Thu 10 Jul | ☐ | L01, L02 | Q1-Q5 (___/5) |
| 3 | Fri 11 Jul | ☐ | L03, L04 | Q6-Q10 (___/5) |
| 4 | Sat 12 Jul | ☐ | L05, L06 | Q11-Q12 (___/2) |
| 5 | Sun 13 Jul | ☐ | L07, L08 | Q13-Q16 (___/4) |
| 6 | Mon 14 Jul | ☐ | L09, L10 | Q17-Q18 (___/2) |
| 7 | Tue 15 Jul | ☐ | L11, L12 | Q19-Q24 (___/6) |
| 8 | Wed 16 Jul | ☐ | study_notes weak areas | Full mock ___/24 |
| 9 | Thu 17 Jul | ☐ | Cheat sheet + Quick Ref sections | Sample paper recall |

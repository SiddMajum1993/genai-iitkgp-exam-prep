# Practice mocks

Eight timed mock exams in the real paper's format: **24 multiple-choice questions, 4 marks each, 90 minutes, no negative marking.**

**Mocks 1–4 are the official practice set.** IIT Kharagpur's 92 practice questions, split into four papers with an even spread of topics in each.

**Mocks 5–8 are original, and each one is different on purpose.** Last time, mocks that copied the sample paper's style made the real exam feel unfamiliar. So these keep the official set's topics and level but change everything else: harder distractors, calculations, scenarios and unfamiliar question forms. None of them shares the official set's habits (the right answer being the longest option, or the same letter over and over).

## What we learned from the official practice set

**It's almost all theory.** Only 2 of its 92 questions need arithmetic. But students from the first exam slot reported formula questions from the fine-tuning chapters, so Mocks 5–8 include more calculations, and Mock 6 is mostly calculations.

**The three blocks are written differently.** The agents block is generic agent theory with easy wrong options. The VLM block tests exact facts from the slides (595K and 158K samples, 256 Qwen-VL tokens, SAM's 32 × 32 grid, ColPali's 0.39 s vs 7.22 s). The fine-tuning block mixes decisions, mechanisms and a couple of calculations.

**It leaves a lot out.** It has no LangChain or LangGraph, no MCP or A2A, almost nothing on multi-agent patterns, Responsible AI frameworks, PPO's four models, DPO details, SLMs or red-teaming. The real exam may still ask about them, so Mocks 7 and 8 cover them.

## The mocks

**[Mock 1](mock-1.md), [Mock 2](mock-2.md), [Mock 3](mock-3.md) and [Mock 4](mock-4.md): the official practice set.** Four parts of 24 questions each, with the options reshuffled and explanations for every option.

**[Mock 5](mock-5.md): same topics, sharper distractors.** The official set's core topics from new angles, with no obviously silly wrong answers.

**[Mock 6](mock-6.md): numbers and formulas.** About a third of the paper is calculation. Revise [Numbers & formulas](../extras/formulas.md) first.

**[Mock 7](mock-7.md): scenarios and design choices.** Short case studies, plus a few config and code snippets to read.

**[Mock 8](mock-8.md): curveballs.** NOT/EXCEPT, two-statement, ordering and "most precise answer" questions on the under-tested topics. Do this one last.

!!! tip "Suggested order"
    Alternate official and original papers so you get used to both styles: **1 → 5 → 2 → 6 → 3 → 7 → 4 → 8**.

## How to take a mock

1. **Set aside 90 minutes** with no notes. That's under 4 minutes a question.
2. **Answer every question.** There's no negative marking, so a blank is always worse than a guess.
3. **Use scratch paper for calculations.** Write the units down.
4. **Flag hard questions** and come back to them after the last one.

## After each mock

**Right and confident:** move on.

**Right, but guessed:** read the explanation anyway. It's a gap that happened to score.

**Wrong:** read the explanation, then reread the chapter section it points to before your next mock.

Keep a note of your weakest section and spend your next study session there.

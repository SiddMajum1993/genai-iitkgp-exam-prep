# Lecture 7 — Advanced Prompt Engineering
**Estimated read time:** 25–30 min · **Difficulty:** Core (very high-yield for exam)

## What this lecture is about

This lecture teaches you how to make large language models solve problems more accurately by changing the way you ask questions. You already know how to use ChatGPT casually, but this lecture reveals the systematic engineering methods behind prompting techniques like zero-shot, few-shot, Chain-of-Thought, ReAct, and Tree-of-Thoughts. These are not just buzzwords. They are proven patterns that researchers use to push models from 40% accuracy to 90% accuracy on reasoning tasks without retraining the model.

The material is structured around the insight that LLMs are autoregressive text predictors, not magic oracles. They predict one token at a time based on everything that came before. By structuring your prompt correctly, you change what the model "sees" when it generates the next token, which changes what it outputs. This lecture shows you exactly how to structure those prompts. Exam questions 13 and others directly test your ability to recognize these patterns, so understanding this lecture is high priority.

## Prerequisites

You should understand what a language model is and what a token is. You do not need to know the mathematics of transformers. If you have used ChatGPT and asked it to do something like "Summarize this text" or "Classify this sentence," you are ready.

---

## Concept 1: Prompting as Programming

### The problem it solves

Traditionally, if you wanted a computer to do something, you wrote code in Python or Java. With large language models, you write instructions in natural language. But unstructured instructions lead to unpredictable outputs. Prompt engineering gives you a structured way to "program" an LLM so it behaves consistently.

### The intuition

Think of the LLM as a very smart intern. If you say "Write me a summary," the intern might write 10 pages or 10 words. If you say "Write me a 3-sentence summary of this article focusing on the financial impact," you get what you want. Prompting is about learning how to give clear, structured instructions so the model outputs what you need.

### What it is

A **prompt** is the initial text you give to the model. It can be a question, a command, a snippet of code, or even a complex paragraph with examples. The prompt is your control interface. The model uses the prompt to generate a response token by token.

There are two types of prompts in real applications:

1. **System prompts**: These are instructions set by the developer. For example, "You are a customer support assistant. Be polite and concise." The user never sees these.
2. **User prompts**: These are what the end user types. For example, "Why is my package late?"

### Why the term "engineering"

Prompting is called engineering because it involves systematic design, iterative refinement, and optimization. You do not just guess at prompts. You test them, measure results, and improve them. Just like software engineering.

### Exam angle

You will be asked to distinguish between different prompting techniques. Know that prompting is not random trial and error. It is a structured process.

---

## Concept 2: Tokens and Decoding Hyperparameters

### The problem it solves

Before we dive into prompting techniques, you need to understand how the model generates text and how you control that process. This is the foundation for everything else.

### The intuition

An LLM does not output whole sentences at once. It generates one token at a time. A token is roughly a word, part of a word, or punctuation. For example, "The cat sat on the mat" might be tokenized as ["The", " cat", " sat", " on", " the", " mat"]. The model looks at all previous tokens and predicts the next one.

The model does not always pick the most likely token. You can control how "creative" or "safe" the model is by adjusting decoding hyperparameters.

### Temperature

Temperature controls the randomness of predictions. It is a positive number.

- **Low temperature (close to 0)**: The model becomes deterministic. It always picks the most probable token. Output is repetitive and safe.
- **High temperature (e.g., 2)**: The model becomes more random. It picks from a wider range of tokens. Output is more creative but less coherent.

**Mathematical intuition:** Temperature divides the logits (raw scores) before applying softmax. Lower temperature makes the probability distribution sharper (peaky). Higher temperature flattens it.

Example: If the model assigns probabilities [0.979, 0.018, 0.002, 0.0003] to ["cat", "dog", "fish", "banana"], then at temperature 0.5, the distribution becomes even more concentrated on "cat" (0.979 → 0.979+). At temperature 2, the distribution flattens, so "dog" and "fish" have a higher chance of being selected.

**Practical rule:** Use low temperature for factual tasks (summarization, classification). Use high temperature for creative tasks (story generation, brainstorming).

### Top-k sampling

Top-k sampling restricts the model to sample from only the top k most likely tokens.

Example: If k=5 and the model's top predictions are ["east": 0.19, "morning": 0.17, "sky": 0.15, "vast": 0.14, "glance": 0.1, ...], the model only considers these 5 tokens. It ignores "glance" and everything below if k=4.

**Why this helps:** It prevents the model from occasionally picking a very improbable token that makes the output nonsensical.

### Top-p (nucleus sampling)

Top-p sampling is dynamic. Instead of fixing the number of tokens (k), you fix the cumulative probability mass (p).

Example: If p=0.65, the model adds up probabilities from the top tokens until the sum exceeds 0.65. Suppose ["east": 0.19, "morning": 0.17, "sky": 0.15, "vast": 0.14] sum to 0.65. The model samples from these 4 tokens only.

**Why this is better than top-k:** When the probability distribution is very sharp (one token has 0.9 probability), top-k still forces the model to consider k tokens even if the others are garbage. Top-p adapts. If one token is overwhelmingly likely, it samples from just that one. If the distribution is flat, it samples from more tokens.

### Exam angle

You may be asked to explain how temperature, top-k, and top-p affect generation. Remember: temperature controls sharpness, top-k fixes the candidate set size, top-p adapts the candidate set based on probability mass.

---

## Concept 3: Zero-Shot Prompting

### The problem it solves

You want the model to perform a task without giving it any examples. This is the simplest prompting technique.

### The intuition

You trust that the model has learned enough from its training data to generalize to your task. You just tell it what to do.

### Worked example

**Prompt:**
```
Classify the text into neutral, negative, or positive. Just give me the classification.
Text: I think the vacation is okay.
Sentiment:
```

**Output:**
```
Neutral
```

Notice: no examples. You just gave an instruction and the input. The model inferred the task from the instruction.

### When it works

Zero-shot prompting works well on:
- Tasks the model has seen many times during training (sentiment analysis, translation, simple Q&A).
- Larger models (e.g., Mistral 123B is better at zero-shot than Mistral 7B).

### When it fails

Zero-shot prompting struggles with:
- Complex reasoning tasks (e.g., multi-step math problems).
- Tasks where the model needs to know your preferred output format.
- Ambiguous instructions.

**Example of failure:**

**Prompt:**
```
Solve the math problem. What is 3/7 of 84?
```

**Model output:**
```
To find 3/7 of 84, we multiply (3/7) * 84.
First, multiply the numerators: 3 * 84 = 252.
Next, multiply the denominators: 7 * 84 = 588.
Now divide: 252 / 588 = 0.428571.
So, 3/7 of 84 is approximately 0.4286.
```

**Wrong.** The correct answer is 36 (because 3/7 * 84 = 3 * 12 = 36). The model confused the procedure.

### How to fix it

Use few-shot prompting (show examples) or Chain-of-Thought prompting (show reasoning).

### Exam angle

Recognize zero-shot prompts by the absence of examples. Expect questions testing whether a given prompt is zero-shot or few-shot.

---

## Concept 4: Few-Shot Prompting

### The problem it solves

The model does not know what output format you want, or it fails at the task when given zero-shot instructions. By showing the model a few input-output examples, you teach it the task.

### The intuition

Think of this as showing a child how to solve a problem by doing a few examples on the board. You do not explain the rules. You show examples, and the child infers the pattern.

### Worked example

**Prompt:**
```
This is awesome // Positive
This is bad! // Negative
Wow that movie was rad! // Positive
What a horrible show! //
```

**Output:**
```
Negative
```

Notice: You gave 3 examples (input // output). The model inferred the pattern. You did not say "Classify these sentences as positive or negative." The model figured that out from the examples.

### The format

Few-shot prompting typically follows this structure:

```
[example 1 input] // [example 1 output]
[example 2 input] // [example 2 output]
...
[test input] //
```

You can use any separator (//,  ->, \n\n). The model is smart enough to infer structure.

### When it works

Few-shot prompting works well for:
- Classification tasks.
- Format-following tasks (e.g., generating structured JSON).
- Tasks where you want a specific style or tone.

### When it fails

Few-shot prompting still struggles with complex reasoning. Here is an example from the lecture:

**Prompt:**
```
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: The answer is False.

The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: The answer is True.

The odd numbers in this group add up to an even number: 16, 11, 14, 4, 8, 13, 24.
A: The answer is True.

The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4, 2.
A: The answer is False.

The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:
```

**Model output:**
```
The answer is True.
```

**Wrong.** The odd numbers are 15, 5, 13, 7, 1. Their sum is 41 (odd). So the answer should be False. The model got the final answer wrong even though it saw 4 examples.

### Why it failed

The model saw the pattern (input → True/False), but it did not learn *how to reason*. The examples did not show the intermediate reasoning steps. The model just saw final answers.

### How to fix it

Use **Chain-of-Thought prompting** (show the reasoning, not just the answer).

### Exam angle (CRITICAL)

You will be tested on the difference between **few-shot prompting** and **few-shot Chain-of-Thought prompting**. In few-shot prompting, you show input-output pairs with NO intermediate reasoning. In few-shot CoT, you show input-reasoning-output triplets. If the examples include reasoning steps, it is CoT. If they do not, it is standard few-shot.

---

## Concept 5: Chain-of-Thought (CoT) Prompting

### The problem it solves

Complex reasoning tasks require breaking the problem into steps. If you only show the final answer, the model does not learn the reasoning process. CoT prompting forces the model to generate intermediate reasoning steps, which improves accuracy on hard problems.

### The intuition

Instead of asking the model to jump straight to the answer, you ask it to "think out loud." You show it how to break the problem into steps. This is like showing your work in a math exam. The process of writing the steps helps you avoid mistakes.

### The key phrase

Chain-of-Thought prompting is triggered by phrases like:
- "Let's think step by step."
- "Let's break this down."
- "Show your reasoning."

### Worked example (Few-shot CoT)

This is the exact example from the sample paper (Question 13).

**Prompt:**
```
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.

The odd numbers in this group add up to an even number: 16, 11, 14, 4, 8, 13, 24.
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.

The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4, 2.
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.

The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:
```

**Model output:**
```
The odd numbers in this group are: 15, 5, 13, 7, 1.
Adding these together gives: 15+5+13+7+1=41.
Since 41 is an odd number, the answer is False.
```

**Correct!** Notice what changed: the examples now include the intermediate reasoning ("Adding all the odd numbers (9, 15, 1) gives 25"). The model learned to mimic that reasoning process.

### Side-by-side comparison: Few-shot vs. Few-shot CoT

| **Few-shot (no reasoning shown)** | **Few-shot CoT (reasoning shown)** |
|------------------------------------|------------------------------------|
| Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1. <br> A: The answer is False. | Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1. <br> A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False. |
| Q: The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24. <br> A: The answer is True. | Q: The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24. <br> A: Adding all the odd numbers (17, 19) gives 36. The answer is True. |
| **What the model learns:** The pattern of inputs and final answers. | **What the model learns:** The reasoning process (identify odd numbers → add them → compare). |
| **Accuracy on test question:** Low (model guesses). | **Accuracy on test question:** High (model mimics the reasoning). |

### When to use CoT

Use CoT when:
- The task involves multi-step reasoning (math, logic, planning).
- The task is ambiguous without showing the process.
- You want the model to justify its answer (explainability).

Do not use CoT for simple tasks (sentiment classification, translation). It wastes tokens and time.

### Exam angle

Sample paper Q13 tests this exact distinction. You will be shown a prompt and asked to identify whether it is few-shot or few-shot CoT. Look for the reasoning steps in the examples. If they are present, it is CoT. If not, it is standard few-shot.

---

## Concept 6: Zero-Shot Chain-of-Thought

### The problem it solves

You want the benefits of CoT (step-by-step reasoning) but you do not want to manually write out examples with reasoning steps. Zero-shot CoT lets you trigger reasoning with a single phrase.

### The intuition

Just add "Let's think step by step" to the end of your prompt. The model will generate its own reasoning chain and then output the final answer.

### Worked example

**Zero-shot (no reasoning):**

**Prompt:**
```
A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: The answer (arabic numerals) is
```

**Output:**
```
8
```

**Wrong.** The correct answer is 4.

**Zero-shot CoT (with reasoning):**

**Prompt:**
```
A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step.
```

**Output:**
```
There are 16 balls in total. Half of the balls are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That means that there are 4 blue golf balls.
```

**Correct!** The phrase "Let's think step by step" triggered the model to generate intermediate reasoning.

### How it works

The phrase "Let's think step by step" was seen many times during the model's training. It signals to the model: "This is a reasoning task. Break it down." The model then mimics the reasoning patterns it learned during training.

### Benefits

- Task-agnostic: You do not need to tailor examples to your specific task.
- Simple: Just add one phrase.
- Effective: Works well on arithmetic, logic, and commonsense reasoning.

### Limitations

- Less control: You cannot guide the model's reasoning as precisely as in few-shot CoT (where you show example reasoning steps).
- Still fails on very hard problems: If the model does not know how to solve the problem at all, "Let's think step by step" will not magically teach it.

### Exam angle

Distinguish zero-shot CoT from few-shot CoT. Zero-shot CoT has the phrase "Let's think step by step" and NO examples. Few-shot CoT has examples that include reasoning steps.

---

## Concept 7: Automatic Chain-of-Thought (Auto-CoT)

### The problem it solves

Manually writing reasoning examples for few-shot CoT is tedious. Auto-CoT automates the generation of diverse reasoning examples.

### The intuition

You have a large dataset of questions. You do not want to manually annotate reasoning chains for all of them. Auto-CoT clusters the questions, picks representative questions from each cluster, uses zero-shot CoT to generate reasoning chains for those questions, and then uses those examples for few-shot CoT.

### The process

1. **Clustering:** Group similar questions together (using embeddings or clustering algorithms). This ensures diversity. For example, one cluster might be "word problems about shopping," another might be "word problems about cooking."

2. **Sampling:** From each cluster, pick one representative question. For example, from the shopping cluster, pick "Zoe bought 3 country albums and 5 pop albums. Each album has 3 songs. How many songs did Zoe buy total?"

3. **Reasoning generation:** Use zero-shot CoT (add "Let's think step by step") to generate a reasoning chain for each representative question.

4. **Few-shot prompting:** Use the representative questions and their reasoning chains as few-shot examples for the test question.

### Why this works

By clustering, you ensure that the few-shot examples cover diverse reasoning patterns. By generating the reasoning chains automatically, you avoid manual annotation. The result is comparable or superior performance to hand-crafted few-shot CoT examples.

### Exam angle

You do not need to implement Auto-CoT. Just know that it is a method to automate the creation of CoT examples by clustering questions and using zero-shot CoT to generate reasoning.

---

## Concept 8: Self-Consistency

### The problem it solves

Even with CoT, the model can still make mistakes. Different reasoning paths might lead to different answers. Self-consistency improves robustness by sampling multiple reasoning paths and taking a majority vote on the final answer.

### The intuition

Imagine asking 5 experts to solve a problem independently. Each expert shows their reasoning and gives an answer. If 4 experts say "18" and 1 says "14," you trust the majority. Self-consistency does this with the LLM.

### The process

1. Use CoT prompting (zero-shot or few-shot).
2. Sample multiple completions (e.g., 5 or 10) by using a non-zero temperature or nucleus sampling.
3. Extract the final answer from each completion.
4. Take the majority vote.

### Worked example

**Prompt (with few-shot CoT):**
```
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for $2 per egg. How much does she make every day?
A:
```

**Sample 1 output:**
```
This means she uses 3 + 4 = 7 eggs every day. She sells the remainder for $2 per egg, so in total she sells 7 * $2 = $14 per day. The answer is $14.
```

**Sample 2 output:**
```
She has 16 - 3 - 4 = 9 eggs left. So she makes $2 * 9 = $18 per day. The answer is $18.
```

**Sample 3 output:**
```
This means she sells the remainder for $2 * (16 - 4 - 3) = $26 per day. The answer is $18.
```

Wait, sample 3 computed $26 but then said $18. Let us assume the final line is what the model output as the answer. So we have:

- Sample 1: $14 (wrong reasoning: sold 7 eggs instead of selling 9)
- Sample 2: $18 (correct)
- Sample 3: $18 (correct)

Majority vote: $18.

### Key insight (CRITICAL for exam)

**Self-consistency is an inference-time strategy.** You cannot identify it from looking at a single prompt. You can only identify it if you see the system generate multiple outputs and then aggregate them.

**Exam trap:** You might be shown a prompt and asked "Is this self-consistency?" The answer is: you cannot tell from the prompt alone. Self-consistency refers to the decoding strategy (sample multiple times, then vote). The prompt itself looks the same as any CoT prompt.

In sample paper Q13, one of the options is "Self-consistency." If the question shows you a single prompt (not multiple outputs), then self-consistency is NOT the answer. The question is testing whether you know that self-consistency is about aggregating multiple reasoning paths, not about the prompt structure.

### When to use self-consistency

Use it when:
- Accuracy is critical and you can afford the extra compute (you have to call the model multiple times).
- The task is complex and reasoning paths vary.

Do not use it when:
- Latency matters (it is slower).
- The task is simple (e.g., sentiment classification; majority vote does not help much).

### Exam angle

Self-consistency is NOT identifiable from a prompt. It is a decoding strategy. If you are asked to identify the prompting technique from a prompt example, and one option is "self-consistency," that option is likely wrong unless the question explicitly mentions sampling multiple outputs.

---

## Concept 9: ReAct (Reasoning + Acting)

### The problem it solves

Many real-world tasks require the LLM to interact with external tools (search engines, calculators, databases). ReAct enables the model to interleave reasoning (thinking about what to do next) and acting (calling tools to gather information).

### The intuition

Imagine you are asked: "Which river flows through the city that hosts the Taj Mahal, and where does it originate?" You cannot answer this from memory alone. You need to:

1. Think: "I need to find which city has the Taj Mahal."
2. Act: Search the web for "Taj Mahal location."
3. Observe: "The Taj Mahal is in Agra, on the banks of the Yamuna river."
4. Think: "Now I need to find where the Yamuna originates."
5. Act: Search "Yamuna river origin."
6. Observe: "The Yamuna originates in the Himalayas."
7. Think: "I can now answer."
8. Finish: "The Yamuna river flows through Agra, and it originates in the Himalayas."

ReAct formalizes this Thought → Action → Observation loop.

### The format

A ReAct prompt looks like this:

```
Question: Which river flows through the city that hosts the Taj Mahal, and where does it originate?

Thought 1: I need to find which city has the Taj Mahal.
Action 1: Search[Taj Mahal location]
Observation 1: The Taj Mahal is located in Agra, on the banks of the Yamuna river.

Thought 2: Now I need to find where the Yamuna originates.
Action 2: Search[Yamuna river origin]
Observation 2: The Yamuna originates in the Himalayas at the Yamunotri glacier.

Thought 3: I can now answer the question.
Finish: The Yamuna river flows through Agra, and it originates in the Himalayas.
```

### How it works

The model generates a thought, then an action (e.g., Search[query]). The action is intercepted by an external system (e.g., a script that calls Google Search API). The result (observation) is fed back into the prompt. The model sees the observation and generates the next thought and action. This loop continues until the model says "Finish[answer]."

### Worked example (dynamic Q&A)

**Question:**
```
Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?
```

**Iteration 1:**

**Thought 1:** I need to search Apple Remote and find the program it was originally designed to interact with.

**Action 1:** Search[Apple Remote]

**Observation 1:** The Apple Remote is a remote control introduced in October 2005 by Apple, originally designed to control the Front Row media center program…

**Iteration 2:**

**Thought 2:** Apple Remote was originally designed to control the Front Row media center program. I need to search Front Row next and find what other device can control it.

**Action 2:** Search[Front Row]

**Observation 2:** Could not find [Front Row]. Similar: ['Front Row Seat to Earth', 'Front Row Motorsports', 'Front Row (software)', …]

**Iteration 3:**

**Thought 3:** Front Row is not found. I need to search Front Row (software).

**Action 3:** Search[Front Row (software)]

**Observation 3:** Front Row is a discontinued media center software…

**Iteration 4:**

**Thought 4:** Front Row (software) is controlled by an Apple Remote or the keyboard function keys. So the answer is keyboard function keys.

**Action 4:** Finish[keyboard function keys]

### When to use ReAct

Use ReAct when:
- The task requires external information (fact-checking, real-time data, knowledge retrieval).
- The problem is multi-step and involves exploration (e.g., navigating a website, debugging code).

Do not use ReAct for:
- Simple closed-domain tasks (e.g., "What is 2+2?").
- Tasks where all information is in the prompt already.

### Exam angle

Recognize ReAct by the Thought → Action → Observation cycle. If you see a prompt with interleaved reasoning and tool calls (Search[...], Lookup[...], etc.), it is ReAct.

---

## Concept 10: Tree-of-Thoughts (ToT)

### The problem it solves

Chain-of-Thought generates one reasoning path. But what if the first path you try is a dead end? Tree-of-Thoughts explores multiple reasoning branches, evaluates them, and backtracks if needed.

### The intuition

Think of solving a puzzle. You try one approach. If it does not work, you backtrack and try another. ToT formalizes this. The model generates multiple possible next steps, evaluates which ones are promising, and explores the best ones further. It is like a decision tree.

### The structure

Instead of a linear chain (Thought 1 → Thought 2 → Thought 3), you have a tree:

```
[Problem]
   ├─ Approach 1 (confidence: Medium)
   │     ├─ Sub-approach 1a (confidence: Low)
   │     └─ Sub-approach 1b (confidence: High) → [Selected] → [Detailed Solution]
   ├─ Approach 2 (confidence: High)
   └─ Approach 3 (confidence: Low)
```

The model generates multiple approaches, evaluates each one (self-evaluation), and then explores the highest-confidence branch.

### Worked example (the ball puzzle)

**Puzzle:**
```
Bob is in the living room. He walks to the kitchen, carrying a cup. He puts a ball in the cup, turns the cup upside down, then carries the cup to the bedroom. Where is the ball?
```

**Usual zero-shot output:**
```
The ball is in the bedroom.
```

(This is plausible but ignores the physics of turning the cup upside down.)

**Tree-of-Thoughts approach:**

**Prompt:**
```
Imagine three different experts are answering this question. They will track the ball step-by-step, evaluate the physics of each action, and branch out to cover different possible interpretations.

Puzzle: Bob is in the living room. He walks to the kitchen, carrying a cup. He puts a ball in the cup, turns the cup upside down, then carries the cup to the bedroom. Where is the ball?
```

**Expert 1 (Physics):**
Cup upside down, gravity makes ball fall out in kitchen.
Conclusion: Kitchen floor.

**Expert 2 (Language):**
Story never says ball stayed inside the cup. "Upside down" implies the ball dropped before moving rooms.
Conclusion: Kitchen.

**Expert 3 (Skeptic):**
The text does not specify the cup type or the ball size. If the ball is small and lodged in the rim, stuck by suction, or the cup has a hidden lid, it could remain inside while Bob carries it to the bedroom. The narrative leaves that possibility open.
Conclusion: Bedroom (ball still in cup).

**Final consensus (majority vote):**
The ball is in the kitchen.

### When it works

ToT is ideal for:
- Puzzles and planning tasks where dead ends are common.
- Creative writing (exploring different plot directions).
- Math optimization (trying different solution strategies).

### Limitations

- Expensive: Generates many reasoning paths. Token cost and latency are high.
- Overkill for simple tasks: If the problem has one clear path, ToT wastes resources.

### Exam angle

Recognize ToT by:
- Multiple parallel reasoning branches (Approach 1, Approach 2, Approach 3).
- Evaluation of branches (confidence scores, selection of best branch).
- Backtracking or pruning of low-confidence branches.

---

## Concept 11: Meta Prompting

### The problem it solves

Instead of solving the problem directly, you ask the model to design the best problem-solving strategy first.

### The intuition

Meta prompting is like asking a teacher: "Before you solve this math problem, tell me the best way to approach it." The model generates a structured solution plan, then executes it.

### The format

A meta prompt might look like:

**Problem Statement:**
```
Problem: [user's question]

Solution Structure:
1. Begin the response with "Let's think step by step."
2. Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.
3. End the solution with the final answer encapsulated in a LaTeX-formatted box.
4. Finally, state "The answer is [final answer to the problem]."
```

Then you give a few-shot example showing this structure, and finally the test problem.

### Why it works

By focusing on the structure of the solution (not the content), you make the model think about how to think. This improves consistency and reasoning quality, especially on complex tasks.

### When to use meta prompting

Use it when:
- The task is complex and benefits from explicit structure (e.g., step-by-step solutions with explanations).
- You want the model to output in a specific format (e.g., LaTeX notation).

### Exam angle

Meta prompting emphasizes structure and syntax over content. If you see a prompt that specifies how the model should structure its response (e.g., "Start with X, then Y, then end with Z"), that is meta prompting.

---

## Concept 12: Directional Stimulus Prompting (DSP)

### The problem it solves

You want to guide the model toward relevant content without giving it full examples. DSP generates hints (keywords, topics, clues) that steer the model in the right direction.

### The intuition

Imagine you are summarizing a long article. Instead of giving the model the full article and saying "Summarize this," you first extract keywords (e.g., "Lee Wan-koo, resign, South Korean tycoon, investigation, notes, top officials"). Then you give those keywords to the model as hints. The model uses the hints to focus on the important parts.

### How it works

1. Train a small policy model (e.g., T5) to generate "directional stimuli" (keywords, key phrases).
2. Given an input (e.g., a news article), the policy model generates hints (e.g., "Lee Wan-koo, resign, scandal").
3. Append the hints to the prompt.
4. The large LLM uses the hints to generate the output (e.g., summary).

### Why it works

The hints act as a filter. They tell the model: "Focus on these aspects." This improves accuracy and reduces hallucination, because the model is less likely to drift into irrelevant details.

### Exam angle

DSP is an advanced technique. You do not need to know the training details. Just know: DSP uses keywords or hints to guide the model. If you see a prompt that includes keywords or clues (e.g., "Keywords: X, Y, Z. Now summarize."), that is directional stimulus prompting.

---

## Concept 13: Prompt Chaining

### The problem it solves

Some tasks are too complex to handle in one prompt. Prompt chaining breaks the task into a sequence of smaller prompts, where the output of one prompt becomes the input to the next.

### The intuition

Instead of saying "Translate this Spanish text to English, extract all statistics, and translate them back to Spanish," you break it into 5 steps:

1. Read the Spanish text.
2. Translate it to English.
3. Extract statistics from the English text.
4. Create a bullet list of the statistics.
5. Translate the bullet list back to Spanish.

Each step is a separate prompt. The output of step 2 is the input to step 3, and so on.

### Worked example

**Single complex prompt (fails):**
```
Consider the given text in Spanish. Translate it into English. Find all the statistics and facts used in this text and list them as bullet points. Translate them again into Spanish.

[Spanish text]
```

This is too complex. The model might lose track of what it is doing.

**Chained prompts (succeeds):**

**Prompt 1:**
```
Translate the following Spanish text into English:
[Spanish text]
```

**Output 1:** [English text]

**Prompt 2:**
```
Extract all statistics and facts from the following text:
[English text]
```

**Output 2:** [List of facts]

**Prompt 3:**
```
Translate the following English facts into Spanish:
[List of facts]
```

**Output 3:** [Spanish facts]

### When to use prompt chaining

Use it when:
- The task has sequential steps.
- Each step's output is verifiable.
- You want intermediate checkpoints (for debugging or audit).

Do not use it when:
- Latency is critical (multiple calls slow things down).
- The steps are not cleanly separable.

### Exam angle

Recognize prompt chaining by multiple sequential prompts, where each prompt takes the previous output as input.

---

## Concept 14: Hallucination

### The problem it solves

LLMs sometimes generate plausible-sounding but incorrect information. This is called hallucination. It is a major limitation.

### The intuition

The model is trained to predict the next token based on patterns in its training data. If you ask it a question it does not know, it will still generate an answer that looks convincing, because that is what it learned to do. It does not say "I don't know."

### Examples of hallucination

**Example 1:**

**Question:**
```
What is the natural habitat of the Elysian Phoenix butterfly?
```

**Model answer:**
```
The Elysian Phoenix butterfly (Elymnias hypermnestra) is a species of brush-footed butterfly found in the Indo-Australian region.
```

**Truth:** The "Elysian Phoenix butterfly" does not exist. The model fabricated the species name and habitat.

**Example 2:**

**Question:**
```
Tell me about the 2007 earthquake in San Diego.
```

**Model answer:**
```
The 2007 San Diego earthquake, also known as the "Borrego Mountain earthquake," occurred on April 4, 2007, at 10:43 AM local time.
```

**Truth:** The Borrego Mountain earthquake happened in 1968, not 2007. There was no major earthquake in San Diego in 2007.

### Why hallucination happens

1. The model learned to be confident and fluent. Saying "I don't know" is rare in the training data.
2. The model does not have access to a database of facts. It only has statistical patterns.
3. Rare facts (e.g., obscure species, minor historical events) are underrepresented in training data, so the model guesses.

### Mitigation strategies

1. **Use retrieval-augmented generation (RAG):** Give the model access to a knowledge base. Before answering, the model retrieves relevant documents and uses them to ground the answer.

2. **Use fact-checking tools:** After the model generates an answer, check it against a trusted source.

3. **Prompt the model to admit uncertainty:** Add to the prompt: "If you don't know, say 'I don't know.'"

4. **Use citations:** Ask the model to cite sources. If it cannot cite a source, flag the answer as potentially hallucinated.

### Exam angle

Know what hallucination is (generating false information presented as fact). Know why it happens (statistical patterns, not knowledge retrieval). Know basic mitigations (RAG, fact-checking, prompting for uncertainty).

---

## Putting it together

You have learned 10+ prompting techniques. Here is how to decide which one to use:

| **Task** | **Technique** | **Why** |
|----------|--------------|---------|
| Simple classification (e.g., sentiment) | Zero-shot or few-shot | Fast, no reasoning needed. |
| Multi-step math or logic | Few-shot CoT or zero-shot CoT | Model needs to show reasoning steps. |
| High-stakes accuracy | Self-consistency | Aggregate multiple reasoning paths. |
| Need external information | ReAct | Model calls tools (search, APIs). |
| Puzzle with multiple solution paths | Tree-of-Thoughts | Explore branches, backtrack. |
| Complex task with sequential steps | Prompt chaining | Break into smaller, verifiable steps. |
| Need specific output format | Meta prompting or few-shot | Show the model the structure. |

**Exam strategy:** You will be shown a prompt and asked to identify the technique. Look for these patterns:

1. **No examples, just instruction** → Zero-shot.
2. **Examples, no reasoning** → Few-shot.
3. **Examples with reasoning** → Few-shot CoT.
4. **"Let's think step by step", no examples** → Zero-shot CoT.
5. **Multiple outputs, majority vote** → Self-consistency (but this is NOT visible in the prompt itself; it is an inference strategy).
6. **Thought → Action → Observation** → ReAct.
7. **Multiple parallel approaches, evaluation** → Tree-of-Thoughts.
8. **Output of prompt 1 feeds into prompt 2** → Prompt chaining.

---

## Check yourself

1. What is the difference between zero-shot and few-shot prompting?
2. You are shown this prompt: "Classify this review: 'The food was cold.' Positive or Negative?" Is this zero-shot or few-shot?
3. You are shown this prompt: "Q: 2+2. A: 4. Q: 3+5. A: 8. Q: 7+1. A:" Is this few-shot or few-shot CoT?
4. You add "Let's think step by step" to a prompt with no examples. What technique is this?
5. You sample 10 outputs from a CoT prompt and take the majority vote on the final answer. What technique is this?

**Answers:**

1. Zero-shot gives no examples; few-shot gives input-output examples.
2. Zero-shot (no examples).
3. Few-shot (no reasoning shown; only final answers).
4. Zero-shot CoT.
5. Self-consistency.

---

## Quick reference (for revision)

- **Zero-shot:** Instruction only, no examples. Works for simple tasks. Fails on complex reasoning.
- **Few-shot:** Instruction + examples (input → output). No reasoning shown. Works for format learning. Fails on reasoning.
- **CoT (Chain-of-Thought):** Show reasoning steps in examples (input → reasoning → output). Works for multi-step tasks.
- **Zero-shot CoT:** Add "Let's think step by step" to trigger reasoning. No examples needed.
- **Auto-CoT:** Automatically generate CoT examples by clustering questions and using zero-shot CoT.
- **Self-consistency:** Sample multiple CoT outputs, majority vote. Improves robustness. NOT visible in the prompt.
- **ReAct:** Thought → Action → Observation loop. For tasks needing external tools.
- **Tree-of-Thoughts:** Explore multiple reasoning branches. Evaluate and backtrack. For puzzles and planning.
- **Meta prompting:** Ask the model to design the solution strategy first. Focus on structure.
- **Directional Stimulus Prompting:** Use hints/keywords to guide the model.
- **Prompt chaining:** Break task into sequential prompts. Output of one is input to next.

**Sample paper Q13 key insight:** Few-shot CoT includes reasoning in the examples. Few-shot does not. Look for "Adding all the odd numbers..." in the examples. If present, it is CoT. If absent, it is few-shot. Self-consistency is NOT identifiable from the prompt alone; it is a decoding strategy.

# 13 · Responsible & trusted AI — governing the AI you already use

**Session:** Module 5, Session 1 (date not stated in the course files) · Facilitator: Sundar Narayanan, AI Ethics Researcher and Advisor · *Responsible & Trusted AI Essentials*
**Read time:** 25–35 min · **Builds on:** Chapter 09 (grounding and citations), Chapters 10–12 (agents that take actions)

!!! warning "How this chapter was built"
    This session had **no slides, no post-read and no Q&A sheet**, only an **agenda** and a **participant pre-read**. Everything that comes from those two documents is presented as course content. Anything added from public sources to fill gaps is marked with a **"Beyond the course slides"** box. Lean hardest on the course-content parts.

---

## In one minute

AI no longer just answers questions. It talks to customers, drafts reports and, more and more, **takes actions** on the organisation's behalf. When it produces fluent, confident output that is **wrong**, customers, courts and regulators hold the **organisation** responsible, not the tool and not the vendor.

So responsible AI is a **management and accountability** issue, not only a technical one. The session gives you:

- **Responsible vs trustworthy AI:** *what we promise* vs *what we can prove*.
- **Five questions** every leader should ask about any AI system. A good answer is **evidence**, not "we believe so".
- **The limits of existing rules**, and the fact that most organisations are **deployers**, not providers, and still answer for the outcome.
- **Five essential controls:** guardrails, privacy, hallucination management, cost control and system documentation.
- **Three 90-day priorities** to take back to your team.

If you remember one sentence, make it this one: **a control is something the system enforces, not something it is asked to do.** An instruction is not a permission, and an alert is not a cap.

---

## Exam radar

!!! caution "Question style is less predictable here"
    There are no class quizzes or Q&A to tell us how this topic gets tested. The content is governance-oriented, so expect **scenario questions** ("which control would have prevented this?", "who is accountable?") rather than technical detail. Likelihoods below are inferred from the agenda and pre-read only.

### 🔴 High: expect questions here

**The five essential controls and the one-line rule for each.** They're the centrepiece of both the agenda and the pre-read, and they fit scenario questions very well.

**Matching a real case to the control that would have helped.** The pre-read pairs every case with a control.

**Responsible vs trustworthy AI ("promise vs proof").** Both the agenda and the pre-read define this.

**The five leader questions, and what good evidence looks like.** These come from a pre-read table.

**Provider vs deployer.** The organisation is accountable even when a vendor or chatbot is at fault. This point is repeated several times, and the Air Canada case illustrates it.

### 🟠 Medium: likely to show up in one question

**The agentic shift.** AI moves from *saying* something wrong to *doing* something wrong. It's shown by the PocketOS case and by the 90-day priority on write/modify/approve/spend authority.

**The three 90-day priorities.** They're listed explicitly in the agenda.

**Governance as one integrated system.** Data, models, tools, human oversight and evidence are governed together. This is an agenda item, only briefly described.

**Framework names and their one-liners.** NIST AI RMF, the EU AI Act, ISO/IEC 42001, the DPDP Act and the India AI Governance Guidelines are listed as optional reading.

### 🟢 Low: understand the idea, skip the details

**Deeper framework details, case figures and dates.** Examples are the EU AI Act tiers and the DPDP roles. They're either not in the course files or peripheral.

---

## Concept 1 — Why responsible AI is a management issue

### The idea in plain words

An AI system can produce output that **sounds** perfectly right and is completely wrong. When that happens, nobody blames "the model". They blame the **company that put it in front of customers**.

The best-known example is in the pre-read.

!!! example "Case: Moffatt v. Air Canada (2024)"
    Air Canada's website chatbot told a grieving customer he could claim a **bereavement refund** after booking. That wasn't the airline's actual policy. Air Canada argued, in effect, that the chatbot was responsible for its own words. The tribunal disagreed and held **Air Canada liable**: *"It makes no difference whether the information comes from a static page or a chatbot."*

### Why it matters

- **System output ≠ organisational accountability.** The output came from a system, but the accountability stays with the organisation.
- Responsibility can't be handed off to a tool or a vendor.

!!! example "Everyday analogy"
    If a shop assistant you hired gives a customer wrong information, the shop is answerable, not the assistant's training college. An AI chatbot is in the same position as that assistant.

---

## Concept 2 — Responsible AI vs trustworthy AI: promise vs proof

**Responsible AI is the promise.** It's the set of obligations and responsibilities an organisation **accepts**. In short, *what we promise to do*.

**Trustworthy AI is the proof.** It's how far the system's behaviour and controls can be **independently examined and substantiated**. In short, *what others can check and confirm with evidence*.

To remember it: responsible = **promise**, trustworthy = **proof**.

!!! example "Everyday example"
    A restaurant says "we follow strict hygiene rules". That's **responsible**: it's a promise. A food-safety inspector can walk into the kitchen and see the temperature logs, cleaning records and certificates. That's **trustworthy**: it's proof that someone else can verify.

### Common misunderstanding

- **"They're synonyms."** No. You can make promises (responsible) without being able to prove anything (not yet trustworthy). Trustworthiness needs **evidence**.

---

## Concept 3 — Five questions every leader should ask

The key rule: **a good answer is evidence, not "we believe so".**

**1. Does it work as intended?** This is about performance and validity. Good evidence is a defined purpose, test results against that purpose, and performance monitored in production.

**2. Who could it harm?** This is about fairness and harm. Good evidence means the affected groups are identified, the impact is assessed, and bias and error rates are checked **for each group**.

**3. Whose data does it use?** This is about privacy and data governance. Good evidence shows the data sources, the consent or lawful basis, and what data leaves the organisation.

**4. Can a non-engineer understand why it acted?** This is about transparency and explainability. Good evidence is a plain-language explanation of the inputs, logic and limits, with reasons available for individual decisions.

**5. Who is accountable when it is wrong?** This is about accountability. Good evidence is a **named** owner, an escalation path, and a remedy process for affected people.

!!! example "Case: Amazon's AI recruiting tool (2018)"
    An experimental résumé-scoring tool was scrapped after it was found to **downgrade résumés containing the word "women's"** (as in "women's chess club captain"). It had learned from historical hiring data. This is question 2 in action: overall accuracy can look fine while **one group** is harmed, which is why you check error rates **per group**.

!!! note "Beyond the course slides: fairness, transparency, accountability"
    These five questions line up with principles found in almost every public AI-ethics framework (OECD, NIST, the EU's trustworthy-AI guidelines):

    - **Fairness:** similar people should be treated similarly, and no group should carry an unfair share of errors. Bias usually comes from **training data** that reflects past patterns, as in the Amazon case.
    - **Transparency / explainability:** people affected by a decision should be able to understand, in plain terms, what the system does and why.
    - **Accountability:** a human or organisation, not "the algorithm", owns each outcome and can be asked to fix it.

    The course didn't cover specific fairness metrics or explainability tools (such as SHAP or LIME), so don't expect detailed questions on them.

---

## Concept 4 — The limits of existing rules: deployers still answer

### The idea in plain words

Laws, regulations, standards, guidance and internal incentives each cover **part** of what an AI system does. **None of them, by itself, governs every single action the system takes.** There's a gap between *what the AI actually does* and *what the rules explicitly address*.

**Provider vs deployer:**

**The provider develops the AI system or places it on the market under its own name.** For example, a company that builds and sells a foundation model or an AI product.

**The deployer uses an AI system under its own authority.** For example, a bank that runs a vendor's chatbot on its website.

**Most organisations are deployers, and they still answer for the outcome.**

!!! example "Case: ANI v. OpenAI (Delhi High Court, 2024–26)"
    The Indian news agency ANI sued OpenAI over the use of its content to train AI. According to the pre-read, interim relief was refused and an appeal is pending: **the law is still unsettled.** This shows that you can't always wait for the rules to tell you what's acceptable.

### Common misunderstanding

- **"If we only use a vendor's model, the vendor is responsible."** No. As the deployer, you're accountable for how it's used and what it tells your customers.

---

## Concept 5 — Governance as one integrated system

### The idea

AI governance isn't a set of separate technical fixes. It's **one organisational system** covering:

- **Data:** what goes in and where it comes from.
- **Models:** what's being used and how it behaves.
- **Tools:** what the AI can connect to and act on.
- **Human oversight:** who can review, approve or stop it.
- **Evidence:** records that prove all of the above.

### The agentic shift: from *says* to *does*

Chatbots that **say** something wrong are one problem. AI agents that **do** something wrong are a bigger one. When an AI system **initiates or executes actions** on the organisation's behalf, accountability for those actions must be set up in advance.

!!! example "Case: PocketOS — an AI agent deletes a database (April 2026)"
    During a routine task, an AI coding agent **deleted a production database and its backups in about nine seconds, without asking**. The result was an outage of more than 30 hours. The lesson: any system with authority to **write, modify, approve or spend** needs strict permissions and human approval for risky actions.

This connects directly to the agent chapters (10–12): tool permissions, human-in-the-loop approval and loop limits are governance controls, not just engineering choices.

!!! example "Everyday analogy"
    Giving an agent tool access is like giving a new intern the company card and the keys to the server room. You wouldn't do that on day one without spending limits and someone to sign off on big decisions.

---

## Concept 6 — The five essential controls

This is the centrepiece of the session. Learn the **one-line rule** for each control and the **case** that goes with it.

**1. Guardrails.** These are limits the **system enforces** on what the AI may say or do, and they're **re-tested after every model, prompt or vendor change**. The key rule: *"An instruction ('don't do X') is not a control; a permission ('cannot do X') is."* The case that goes with it is the DPD chatbot.

**2. Privacy.** These are the rules and technical controls on what data goes into AI tools, where it goes, and how long it's kept. The key rule: under India's **DPDP Act**, the organisation stays responsible **even when a vendor processes the data**. The case is the Samsung code leak.

**3. Hallucination management.** A hallucination is fluent, confident output that is **false**: invented facts, clauses or citations. The key rule: treat AI output as a **draft**, and check it **in proportion to the stakes**. The cases are the fake citations at the Supreme Court of India, Deloitte Australia and Air Canada.

**4. Cost control.** This means budgets, **hard spending caps**, loop limits and a **named owner** of the AI bill. The key rule: *"An alert is not a cap."* The case is the US$82,314 Gemini bill run up in 48 hours.

**5. System documentation.** Every significant AI system gets a **one-page system record**. The key rule: it records the **owner**, the **boundaries** (what the system may do and must never do) and **how failures are detected**. The case is the NYC MyCity chatbot.

### The pattern behind the rules

Two rules have the same shape. Spot it and you can answer most control questions:

- **Instruction vs permission.** Telling the model "don't do X" in a prompt can be ignored or bypassed. Making X **impossible** (the tool simply isn't available, or the account lacks the right) is a real control.
- **Alert vs cap.** An email saying "you've spent a lot" doesn't stop the spending. A **hard cap** does.

**Controls must be enforced by the system, not merely requested or observed.**

!!! example "Everyday analogy"
    A sign saying "Please don't enter" is an instruction. A **locked door** is a permission control. A text from your bank saying "large transaction detected" is an alert. A **daily card limit** is a cap.

### The cases, one by one

- **DPD chatbot (2024), guardrails.** After a **system update**, the parcel company's chatbot **swore at a customer** and called DPD the "worst delivery firm". Lesson: re-test guardrails after **every** change.
- **Samsung code leak (2023), privacy.** Engineers pasted **confidential source code** into ChatGPT. Samsung then restricted generative AI on company devices. Lesson: control what data goes into external AI tools.
- **Supreme Court of India on fake AI citations (2026), hallucination.** The court set aside tribunal orders that relied on **non-existent, AI-generated precedents** and declared "zero tolerance". Lesson: verify, especially when the stakes are high.
- **Deloitte Australia (2025), hallucination.** A government report was found to contain **non-existent references**. It was re-issued with disclosure of the AI tools used, and part of the fee was repaid.
- **US$82,314 Gemini bill (2026), cost control.** A small firm, normally spending about US$180 a month, reports that a suspected **stolen API key** ran up a huge bill in two days (the firm's own account; outcome unconfirmed). Lesson: hard caps, key security and a named owner.
- **NYC MyCity chatbot (2024–26), documentation and ownership.** An official city chatbot gave businesses advice that would have them **break the law**, yet it stayed online with **disclaimers**. Lesson: a disclaimer isn't a control. Someone must own the system and its boundaries.
- **Grok image edits on X (Dec 2025–Jan 2026), guardrails and safety.** Users tagged Grok in public replies to digitally "undress" photos of real people without consent. Regulators in India, the UK and the EU moved against the platform. Lesson: harm to **third parties** who never used the tool.

!!! note "Beyond the course slides: how these controls are usually implemented"
    The course gave one-line definitions. Common real-world ways to implement each control:

    - **Guardrails:** input and output filters, **least-privilege** tool permissions, allow-lists of actions, human approval for risky steps, and a regression test suite re-run after every change.
    - **Privacy:** data classification (what may never be sent to an external tool), enterprise contracts with vendors, retention limits, and masking personal data before it's sent.
    - **Hallucination:** grounding with RAG and citations (Chapter 09), human review scaled to risk, and allowing the model to say "I don't know".
    - **Cost:** per-key and per-project hard limits, rate limits, loop or iteration limits for agents, key rotation, and a named budget owner.
    - **Documentation:** practices such as model cards or system cards. The course's version is a simple **one-page record** (owner, boundaries, failure detection); the full template was handed out in class and isn't in the files.

---

## Concept 7 — From failure to action: the 90-day priorities

The session ended with a team exercise. Groups of 5–6 redesigned a realistic AI failure and identified which controls would have changed the outcome.

**Questions to ask about any AI tool in your organisation (from the pre-read):**

1. Who owns it **by name** on the business side (not IT, not the vendor)?
2. What can it do? Only **advise**, or also **write, approve or spend**? Who decided that?
3. **The 2 a.m. test:** if it went wrong at 2 a.m., how would you find out, and who could stop it?

**Three 90-day priorities:**

1. Assign a **clear owner** to every AI use case in production.
2. Identify which systems have **authority to write, modify, approve or spend**.
3. Establish a **concise evidence record** (the one-page system record) for every significant AI system.

---

## Concept 8 — The frameworks and laws named in the pre-read

The pre-read lists these as optional further reading, with one line each. **The course-content line comes first; anything added afterwards is beyond the course.**

**NIST AI Risk Management Framework (USA).** It's voluntary and widely used, with four functions: **Govern, Map, Measure, Manage**. It also has a Generative AI profile.

**EU AI Act.** This is **risk-based** regulation, with **separate duties for providers and deployers**.

**ISO/IEC 42001:2023.** An international, **certifiable** standard for an **AI management system**.

**India's DPDP Act.** The organisation **remains responsible even when a vendor processes the data**.

**India AI Governance Guidelines (MeitY, Nov 2025).** They're **principle-based**, with seven "sutras", and they build on **existing law**.

!!! note "Beyond the course slides: what these frameworks contain"
    **NIST AI RMF 1.0 (2023): the four functions**

    - **Govern:** build the culture, policies, roles and accountability for AI risk. It's **cross-cutting**, so it underpins the other three.
    - **Map:** understand the context: what the system is for, who it affects and what could go wrong.
    - **Measure:** assess and track the identified risks with tests and metrics.
    - **Manage:** prioritise and act on the risks (mitigate, transfer, accept or stop), and keep monitoring.

    A simple way to remember it: *set the rules → find the risks → measure them → act on them.*

    **EU AI Act: four risk levels**

    - **Unacceptable risk, banned.** Examples: government-style social scoring, manipulative systems that exploit vulnerabilities.
    - **High risk, strict obligations.** Examples: AI used in hiring, credit scoring, education, critical infrastructure and law enforcement. Obligations include risk management, data quality, documentation, human oversight and conformity assessment.
    - **Limited risk, transparency duties.** Example: people must be told when they're talking to a chatbot, and AI-generated content such as deepfakes must be disclosed.
    - **Minimal risk, no new obligations.** Examples: spam filters, AI in video games.

    The Act also places separate obligations on providers of **general-purpose AI models**. Duties differ for **providers** and **deployers**, which matches the course's point.

    **ISO/IEC 42001:2023**

    - A **management-system** standard, in the same family style as ISO 27001 for information security.
    - An organisation sets up policies, roles, risk assessment and continual improvement for its AI, and can be **audited and certified** against it.
    - It certifies the **organisation's processes**, not an individual model.

    **India's Digital Personal Data Protection (DPDP) Act, 2023**

    - The **Data Principal** is the person the data is about.
    - The **Data Fiduciary** is the organisation deciding why and how the data is processed. It carries the main obligations: a lawful basis or consent, purpose limitation, security safeguards and breach notification.
    - A **Data Processor** (for example, a vendor) processes data on the fiduciary's behalf. The **fiduciary stays accountable**, which is the course's point.
    - A **Data Protection Board of India** handles enforcement.

    **India AI Governance Guidelines (Nov 2025)**

    - The course names only "seven sutras" (guiding principles).
    - The published guidelines describe themes such as trust, people-first design, fairness, accountability, understandability, and safety and resilience.
    - Check the official MeitY document for the exact wording if you need it.

!!! example "Everyday analogy for the four kinds of rules"
    - **Law** is like traffic law: binding, with penalties (the EU AI Act, the DPDP Act).
    - **Standards** are like a driving-school certification you can earn and be audited against (ISO/IEC 42001).
    - **Guidance and frameworks** are like a good driving handbook: voluntary but widely followed (NIST AI RMF, India's guidelines).
    - **Incentives** are like insurance premiums that reward careful drivers.

    No single one of them covers every situation on the road, and that's the course's "limits of existing rules" point.

---

## Which control when? (scenario helper)

Scenario questions usually hide a clue that points to one control. Here's how to read the most common ones.

**The bot misbehaves after an update or model change → Guardrails.** Guardrails must be re-tested after every change.

**"We told the model in its prompt never to…", and it did anyway → Guardrails.** Turn the instruction into a permission the system enforces.

**Staff paste confidential or personal data into a public AI tool → Privacy.**

**Invented citations, clauses or facts appear in a report or legal filing → Hallucination management.** Treat output as a draft and check it in proportion to the stakes.

**A surprise huge bill, a runaway agent loop or a leaked API key → Cost control.** Use a hard cap, loop limits and a named owner.

**No one knows who owns the bot, and a disclaimer is used instead of a fix → System documentation.** Write the one-page record: owner, boundaries and failure detection.

**An agent deletes, approves or spends without asking → Permissions plus human approval** for write/modify/approve/spend authority. This is guardrails plus governance.

**A chatbot made a promise and the company says "the bot said it, not us" → Accountability.** The organisation is liable, as in the Air Canada case.

---

## Traps to watch for in MCQs

- ❌ "Adding 'never do X' to the system prompt is a guardrail." → It's an **instruction**, not a control. A guardrail **prevents** X.
- ❌ "A spending alert controls cost." → **An alert is not a cap.**
- ❌ "A disclaimer makes a wrong chatbot acceptable." → MyCity shows disclaimers don't replace ownership and boundaries.
- ❌ "The vendor is responsible for its model's mistakes in our product." → The **deployer** still answers for the outcome. Under DPDP, the organisation stays responsible even when a vendor processes the data.
- ❌ "Responsible AI and trustworthy AI mean the same thing." → **Promise vs proof.**
- ❌ "Guardrails are set once at launch." → They must be **re-tested after every model, prompt or vendor change**.
- ❌ "Hallucinated output should always be fully re-checked or never used." → Check it **in proportion to the stakes**, and treat it as a draft.
- ❌ "Existing laws cover everything an AI system does." → None of them, alone, governs every action.
- ❌ "Good evidence = the team believes the system works." → Evidence means **test results, monitoring, named owners and records**.

---

## Recap

1. **The organisation is accountable** for its AI's output and actions, even when the AI or a vendor "made the mistake" (Air Canada).
2. **Responsible = promise; trustworthy = proof.** Answer the five leader questions with **evidence**.
3. **Most organisations are deployers**, and existing rules don't cover every AI action, so build your own controls.
4. **Five controls:** guardrails (permission, not instruction; re-test after changes), privacy, hallucination management (draft + proportional checks), cost control (cap, not alert; named owner), and a one-page system record (owner, boundaries, failure detection).
5. **Agents raise the stakes:** know which systems can write, modify, approve or spend, give every system a named owner, and keep an evidence record.

---

## Practice questions

Pick an option to see instantly whether you're right, with an explanation for every option.

<div class="practice-quiz" data-quiz="ch13">
  <p><em>Loading practice questions…</em></p>
</div>

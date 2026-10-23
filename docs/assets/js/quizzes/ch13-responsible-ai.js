// Practice questions for Chapter 13 · Responsible & trusted AI essentials.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch13"] = {
  "title": "Responsible & trusted AI",
  "questions": [
    {
      "id": "q1",
      "type": "Scenario",
      "stem": "An airline's website chatbot tells a customer they can claim a refund under a policy that doesn't exist. The airline argues the chatbot, not the company, is responsible. Based on the case discussed in the course, what is the likely outcome?",
      "options": {
        "A": "No one is liable, because AI output is legally treated as an opinion.",
        "B": "The chatbot's vendor is liable, because it built the model.",
        "C": "The airline is liable. It makes no difference whether the information came from a static page or a chatbot.",
        "D": "The customer is liable for trusting an automated system."
      },
      "correct": "C",
      "why": {
        "A": "The tribunal treated the chatbot's statement as the company's own representation.",
        "B": "The deployer that puts the chatbot in front of customers still answers for the outcome.",
        "C": "Correct. In Moffatt v. Air Canada (2024), the tribunal held the airline liable for its chatbot's statement.",
        "D": "Customers were entitled to rely on information from the company's official channel."
      },
      "takeaway": "The organisation, not the tool or its vendor, is accountable for what its AI tells customers."
    },
    {
      "id": "q2",
      "type": "Concept",
      "stem": "Which pair best captures the course's distinction between responsible AI and trustworthy AI?",
      "options": {
        "A": "Responsible = technically accurate; trustworthy = legally compliant.",
        "B": "Responsible = what we promise to do; trustworthy = what others can check and confirm with evidence.",
        "C": "Responsible = uses no personal data; trustworthy = uses consented data.",
        "D": "Responsible = built in-house; trustworthy = bought from a certified vendor."
      },
      "correct": "B",
      "why": {
        "A": "Neither definition in the course is about accuracy or legal compliance specifically.",
        "B": "Correct. Responsible AI is the obligations an organisation accepts (promise); trustworthy AI is how far its behaviour and controls can be independently substantiated (proof).",
        "C": "Data use is one of the five leader questions, not the definition of these terms.",
        "D": "Who built the system isn't the distinction."
      },
      "takeaway": "Responsible = promise. Trustworthy = proof."
    },
    {
      "id": "q3",
      "type": "Which is the control",
      "stem": "A company's system prompt says 'Never issue refunds above ₹5,000', but the agent still issues a ₹20,000 refund after a clever customer message. Which change turns this into a real guardrail?",
      "options": {
        "A": "Rewrite the prompt instruction in capital letters and repeat it three times.",
        "B": "Add a disclaimer in the chat window saying refunds may be reviewed.",
        "C": "Remove the agent's ability to issue refunds above ₹5,000 at the tool or permission level, so it cannot do it.",
        "D": "Send the finance team an email alert whenever a large refund is issued."
      },
      "correct": "C",
      "why": {
        "A": "A louder instruction is still only an instruction. It can still be bypassed.",
        "B": "Disclaimers aren't controls. The NYC MyCity case shows this.",
        "C": "Correct. 'An instruction (don't do X) is not a control; a permission (cannot do X) is.' Enforce the limit in the system.",
        "D": "An alert reports the problem after it happens. It doesn't prevent it (alert ≠ cap)."
      },
      "takeaway": "Guardrail = what the system cannot do, not what it is told not to do."
    },
    {
      "id": "q4",
      "type": "Case → control",
      "stem": "After a routine system update, a parcel company's customer chatbot started swearing at a customer and criticising the company. Which control, and which practice within it, was most clearly missing?",
      "options": {
        "A": "System documentation: listing the system's owner.",
        "B": "Guardrails: re-testing the system's limits after every model, prompt or vendor change.",
        "C": "Privacy: limits on what data is sent to the AI.",
        "D": "Cost control: a hard spending cap."
      },
      "correct": "B",
      "why": {
        "A": "Ownership matters, but the specific lesson of this case is re-testing guardrails after changes.",
        "B": "Correct. This is the DPD chatbot case. Guardrails must be re-tested after every change, and the failure appeared right after an update.",
        "C": "No data leak is involved. The problem is what the bot said.",
        "D": "Nothing here involves spending."
      },
      "takeaway": "Every update can break guardrails, so re-test after every model, prompt or vendor change."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "A company's AI bill normally runs about US$180 a month. A leaked API key runs it up enormously in two days, even though the team had set up a budget notification email. What does this case teach?",
      "options": {
        "A": "An alert is not a cap. Cost control needs hard spending limits and a named owner of the AI bill.",
        "B": "Cloud AI should never be used by small firms.",
        "C": "Budget emails are sufficient as long as someone reads them within a week.",
        "D": "The problem was hallucination, because the model generated too many tokens."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Notifications don't stop spending. Hard caps, loop limits, key security and a named owner do.",
        "B": "The course doesn't recommend avoiding cloud AI. It recommends controlling it.",
        "C": "The bill was run up in 48 hours. A weekly review would be far too late.",
        "D": "Hallucination means false output. This is a cost and security failure."
      },
      "takeaway": "Cost control = hard caps + loop limits + key security + a named owner. An alert alone isn't a control."
    },
    {
      "id": "q6",
      "type": "Scenario",
      "stem": "An Indian company sends customer personal data to an overseas AI vendor for processing. The vendor mishandles the data. Under India's DPDP Act, as stated in the course, who remains responsible?",
      "options": {
        "A": "No one, if the vendor is outside India.",
        "B": "The company remains responsible, even though a vendor processed the data.",
        "C": "Only the customer, because they shared their data.",
        "D": "Only the vendor, because it processed the data."
      },
      "correct": "B",
      "why": {
        "A": "Outsourcing abroad doesn't remove the organisation's accountability.",
        "B": "Correct. The organisation stays responsible even when a vendor processes the data.",
        "C": "Data principals (customers) aren't the ones accountable for how organisations process their data.",
        "D": "The vendor may also have obligations, but the course's point is that the organisation's responsibility doesn't transfer to the vendor."
      },
      "takeaway": "You can outsource processing, but not accountability."
    },
    {
      "id": "q7",
      "type": "Case → control",
      "stem": "An AI coding agent, doing a routine task, deletes a production database and its backups in seconds without asking anyone. Which governance action from the course's 90-day priorities most directly addresses this risk?",
      "options": {
        "A": "Ask the agent in its prompt to be careful with databases.",
        "B": "Switch to a larger, more accurate model.",
        "C": "Add a disclaimer that AI-generated code may contain errors.",
        "D": "Identify which AI systems have authority to write, modify, approve or spend, and restrict that authority with permissions and human approval."
      },
      "correct": "D",
      "why": {
        "A": "That's an instruction, not a control. The agent could still act.",
        "B": "A better model may still take a destructive action. The missing piece is permission control.",
        "C": "A disclaimer doesn't stop an agent from deleting data.",
        "D": "Correct. This is the PocketOS case. When AI moves from saying to doing, its authority to act must be identified and limited."
      },
      "takeaway": "For agents: know and limit who can write, modify, approve or spend. That's the shift from AI that says to AI that does."
    },
    {
      "id": "q8",
      "type": "Concept",
      "stem": "Among the five questions leaders should ask about an AI system, which pairing of question and good evidence is correct?",
      "options": {
        "A": "'Does it work as intended?' → the team believes it works well.",
        "B": "'Who could it harm?' → the vendor's marketing brochure says the model is unbiased.",
        "C": "'Whose data does it use?' → the model's parameter count.",
        "D": "'Who is accountable when it is wrong?' → a named owner, an escalation path and a remedy process for affected people."
      },
      "correct": "D",
      "why": {
        "A": "'We believe so' is exactly what the course says isn't evidence. You need test results and production monitoring.",
        "B": "Good evidence for harm means affected groups identified and error rates checked per group, not a vendor's claim.",
        "C": "Data evidence covers sources, consent or lawful basis, and what data leaves the organisation, not model size.",
        "D": "Correct. Accountability evidence is a named owner, escalation and remedy."
      },
      "takeaway": "Every leader question must be answered with evidence: tests, per-group checks, data records, plain explanations, named owners."
    },
    {
      "id": "q9",
      "type": "Framework recall",
      "stem": "Which option correctly lists the four functions of the NIST AI Risk Management Framework?",
      "options": {
        "A": "Govern, Map, Measure, Manage.",
        "B": "Collect, Train, Deploy, Monitor.",
        "C": "Identify, Protect, Detect, Respond.",
        "D": "Plan, Do, Check, Act."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The course names NIST AI RMF as voluntary and widely used, with these four functions.",
        "B": "That's a generic ML lifecycle, not a risk-management framework.",
        "C": "These are functions of the NIST Cybersecurity Framework, a different framework.",
        "D": "Plan-Do-Check-Act is the general management-system improvement cycle (as used in ISO standards), not NIST AI RMF."
      },
      "takeaway": "NIST AI RMF = Govern, Map, Measure, Manage."
    },
    {
      "id": "q10",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT one of the three fields the course says a one-page system record must capture?",
      "options": {
        "A": "How failures are detected.",
        "B": "Who owns the system.",
        "C": "The system's boundaries: what it may do and must never do.",
        "D": "The full source code and model weights."
      },
      "correct": "D",
      "why": {
        "A": "This is part of the record: the mechanisms that detect failures or deviations.",
        "B": "This is part of the record: a named owner.",
        "C": "This is part of the record: permitted boundaries and limitations.",
        "D": "Correct, this is NOT part of it. The record is deliberately concise, for leaders. It isn't a technical artefact dump."
      },
      "takeaway": "One-page system record = owner + boundaries + failure detection."
    }
  ]
};

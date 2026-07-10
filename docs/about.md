# About this site

## What this is

A community-contributed study guide for the **upGrad × IIT Kharagpur *Certificate Course on Generative and Agentic AI***. The course covers 12 lectures on modern LLMs, RAG systems, agentic AI, and safety, culminating in a proctored MCQ exam.

This site distills those 12 lectures into three depths — deep learn-guides, compressed revision notes, and a one-page cheat sheet — plus a full mock exam for practice.

## What this is **not**

- Not an official upGrad or IIT Kharagpur publication
- Not affiliated with, endorsed by, or connected to the course faculty
- Not a substitute for attending the lectures — the guides *summarise and reframe*; they don't replace the original teaching

## Copyright & attribution

- **Lecture slides, sample paper, and course materials** are © upGrad × IIT Kharagpur and their respective authors. None of that content is redistributed on this site. If you're enrolled, download the originals from your course dashboard.
- **Study guides, notes, mock exams, and cheat sheets on this site** are original explanatory works by learners preparing for the same exam, released under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Code** in the repository (site config, extraction scripts, build scripts) is released under the MIT License.

## How to contribute

Corrections, clarifications, and additional practice questions are welcome:

1. Fork the [GitHub repository](https://github.com/your-username/genai-iitkgp-exam-prep)
2. Edit the relevant markdown file in `docs/`
3. Open a pull request describing what you changed and why

If you spot an error but don't want to open a PR, file an issue.

## How to build locally

Clone the repo and:

```bash
pip install -r requirements.txt
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000).

## Site tech

- **[MkDocs Material](https://squidfunk.github.io/mkdocs-material/)** for the static site
- **[MathJax](https://www.mathjax.org/)** for equations
- **[Cloudflare Pages](https://pages.cloudflare.com/)** for hosting
- **GitHub Actions** for auto-deploy on push to `main`

## Feedback

If a concept explanation didn't land for you, that's a documentation bug — [file an issue](https://github.com/your-username/genai-iitkgp-exam-prep/issues) so we can rewrite it more clearly for the next learner.

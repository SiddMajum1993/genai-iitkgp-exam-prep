# GenAI & Agentic AI — Exam Prep

A learner-built study guide for the **upGrad × IIT Kharagpur *Certificate Course on Generative and Agentic AI*** exam.

**Live site:** _(deploy to Cloudflare Pages and add the URL here)_

## What's inside

| Folder | Purpose |
|---|---|
| [`docs/learn/`](docs/learn/) | Deep learn-guides — 800–2,000 lines per lecture, plain English, analogies first |
| [`docs/revise/`](docs/revise/) | Compressed revision notes — formulas, tables, flashcards |
| [`docs/practice/`](docs/practice/) | Full mock exam matching the real paper's format |
| [`docs/cheatsheet.md`](docs/cheatsheet.md) | One-page skim for the day before the exam |
| [`docs/schedule.md`](docs/schedule.md) | Suggested 9-day study plan |
| [`scripts/`](scripts/) | PPTX/PDF extraction helpers (for local content-prep only) |
| `source_local_only/` | Copyrighted source materials (gitignored, never published) |

## About the exam

- 24 MCQs × 4 marks = 96 marks
- 90 minutes
- **No negative marking** — answer every question
- Passing bar is typically 50%; aim for 70% with buffer

## How to use this repo as a learner

1. Read the [study schedule](docs/schedule.md) once. Pick a start date.
2. On study days, work through the day's [learn guides](docs/learn/).
3. On Day 8, take [Mock Exam 1](docs/practice/mock-exam-1.md) under real conditions.
4. On Day 9, read the [cheat sheet](docs/cheatsheet.md) twice. Sleep.
5. Take the real exam.

## How to build the site locally

```bash
# Install dependencies
pip install -r requirements.txt

# Serve locally with hot reload
mkdocs serve

# Build static site to ./site
mkdocs build
```

Then open [http://localhost:8000](http://localhost:8000).

## How to deploy to Cloudflare Pages

The included [GitHub Actions workflow](.github/workflows/deploy.yml) builds and deploys on every push to `main`. You need to set up two things:

1. **Create a Cloudflare Pages project** connected to this GitHub repo (or use direct upload via the wrangler action)
2. **Add these repository secrets in GitHub** → Settings → Secrets and variables → Actions:
    - `CLOUDFLARE_API_TOKEN` — from Cloudflare dashboard → My Profile → API Tokens → "Edit Cloudflare Workers" template
    - `CLOUDFLARE_ACCOUNT_ID` — from Cloudflare dashboard → right sidebar
3. **Set the project name** in `.github/workflows/deploy.yml` (default: `genai-exam-prep`)

Alternatively, connect the repo directly in Cloudflare Pages dashboard and set:
- **Build command:** `pip install -r requirements.txt && mkdocs build`
- **Build output directory:** `site`
- **Environment variables:** `PYTHON_VERSION=3.11`

## Copyright & attribution

- **Lecture slides, sample paper, and course content** are © upGrad × IIT Kharagpur. **Not redistributed here.** If enrolled, download originals from your course dashboard.
- **Study guides, notes, mock exams, cheat sheets** on this site are original explanatory works by learners preparing for the same exam. Released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
- **Code** (site config, scripts, workflows) — MIT License.

## Contributing

Corrections and improvements welcome. Please:

1. Fork the repo
2. Edit the relevant file in `docs/`
3. Preview locally with `mkdocs serve`
4. Open a PR with a description of what changed and why

If you're a learner who's just taken the exam and remembers a question pattern that isn't well-covered here, that's *especially* welcome.

## Disclaimer

This is a community-built study aid. It is **not** affiliated with, endorsed by, or connected to upGrad, IIT Kharagpur, or the course faculty. Use it alongside — not as a replacement for — the official course materials.

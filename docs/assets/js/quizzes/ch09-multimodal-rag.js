// Practice questions for Chapter 09 · Multimodal RAG.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch09"] = {
  "title": "Multimodal RAG",
  "questions": [
    {
      "id": "q1",
      "type": "Scenario",
      "stem": "A bank's RAG assistant answers well from the text of its quarterly reports but keeps failing on questions like 'Which region grew fastest in Q3?', whose answers appear only in bar charts. What is the most likely root cause?",
      "options": {
        "A": "The vector index uses ANN instead of exact search.",
        "B": "The pipeline extracts only text, so the visual information that carries the answer is discarded before retrieval.",
        "C": "The LLM is too small and needs to be fine-tuned on banking data.",
        "D": "The chunk size is too large for the text encoder."
      },
      "correct": "B",
      "why": {
        "A": "ANN costs only a tiny bit of accuracy. It doesn't explain systematic failure on chart-only answers.",
        "B": "Correct. Text-only RAG 'simply cannot see the chart that answers the question'. This is exactly the motivation for multimodal RAG.",
        "C": "A bigger or fine-tuned LLM still can't answer from a chart it never receives. The failure happens before generation.",
        "D": "Chunking can hurt retrieval, but the answer isn't in any text chunk at all, so changing chunk size won't help."
      },
      "takeaway": "When answers live in charts or tables, text-only RAG fails at extraction. Bring in multimodal retrieval and a VLM generator."
    },
    {
      "id": "q2",
      "type": "Concept",
      "stem": "In ColPali-style late interaction, how is a page's relevance score for a query computed?",
      "options": {
        "A": "The whole page is pooled into one vector and compared with one query vector.",
        "B": "The query and page are fed together into a cross-encoder that outputs a single score.",
        "C": "OCR text from the page is compared word-by-word with the query using BM25.",
        "D": "Each query token finds its best-matching patch on the page, and those best-match scores are added up."
      },
      "correct": "D",
      "why": {
        "A": "That's single-vector retrieval. It blurs local details like a small chart, which is what late interaction avoids.",
        "B": "That describes a cross-encoder reranker: more accurate but much slower, and not what late interaction does.",
        "C": "ColPali doesn't use OCR at all. It embeds the page image.",
        "D": "Correct. That's MaxSim: take the max over patches for each query token, then sum across query tokens."
      },
      "takeaway": "Late interaction = many vectors per side; each query token finds its best patch; add the best matches."
    },
    {
      "id": "q3",
      "type": "Which is NOT",
      "stem": "Which of these is NOT a real advantage of ColPali (retrieval in vision space) over a parse-then-embed OCR pipeline?",
      "options": {
        "A": "It can show a heatmap of which page regions matched the query.",
        "B": "Its index is smaller and it needs less GPU memory.",
        "C": "It avoids brittle OCR and layout-analysis steps that lose information on charts and tables.",
        "D": "Offline indexing per page is much faster."
      },
      "correct": "B",
      "why": {
        "A": "This is a real advantage. Patch-level scores can be drawn over the page as an explainability heatmap.",
        "B": "Correct, this is NOT an advantage. With about 1,000 vectors per page and a large VLM backbone, ColPali needs a much larger index and more GPU memory.",
        "C": "This is a real advantage. There's no OCR or layout step to fail.",
        "D": "This is a real advantage. The paper reports roughly 18× faster indexing than the OCR + layout + text-embedding pipeline."
      },
      "takeaway": "ColPali is more accurate on visual pages and faster to index, but heavier to store and run."
    },
    {
      "id": "q4",
      "type": "Scenario",
      "stem": "A fund house wants to pull a fixed set of numbers from thousands of yearly PDF fact sheets. The PDFs are digitally generated with clean, consistent tables and no charts. The budget is tight. What does the course recommend?",
      "options": {
        "A": "A traditional PDF parser feeding text RAG, because it's cheaper and works well for straightforward text and table extraction.",
        "B": "ColPali, because vision-space retrieval is always the most accurate.",
        "C": "Image search with CLIP over the PDF pages.",
        "D": "Fully fine-tune a VLM on every fact sheet."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The Q&A answer: a PDF parser is cheaper and preferred for straightforward text and tables. Use ColPali when visuals matter.",
        "B": "ColPali shines on complex layouts, charts and scans. Here it would add cost without much benefit.",
        "C": "CLIP image search matches overall picture content, not exact table values.",
        "D": "Fine-tuning doesn't supply document facts. That's retrieval's job, and fine-tuning is costly."
      },
      "takeaway": "Choose the strategy by document type: clean text or tables → parser + text RAG; visually complex → vision-space retrieval."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "Why can a CLIP-based image search system match the text query 'snow-capped mountains at sunset' against untagged photos?",
      "options": {
        "A": "Contrastive training placed text and image embeddings in one shared space, so the text vector can be compared directly with image vectors.",
        "B": "It uses OCR to read any text visible in the photos.",
        "C": "It relies on human-added tags stored with each photo.",
        "D": "It first generates a caption for every photo and then runs keyword search over the captions."
      },
      "correct": "A",
      "why": {
        "A": "Correct. CLIP's encoders are trained so matching image–text pairs land close together, enabling direct cosine-similarity search.",
        "B": "OCR reads characters. Mountain photos have no text to read.",
        "C": "The class example stressed that the photos had no captions or tags.",
        "D": "No translation or captioning step is needed. That's the point of a shared space."
      },
      "takeaway": "Shared embedding space = direct text-to-image comparison. Same ANN infrastructure; only what's embedded changes."
    },
    {
      "id": "q6",
      "type": "Symptom → fix",
      "stem": "A multimodal RAG system retrieves the correct page (verified) but the VLM still answers with an outdated figure from its training data. Which change most directly targets this failure?",
      "options": {
        "A": "Add more pages to the index.",
        "B": "Increase the number of patch vectors per page.",
        "C": "Switch the retriever from ColPali to BM25.",
        "D": "Use a grounded prompt: answer only from the retrieved pages, cite the page, and say 'not found' if the answer isn't there."
      },
      "correct": "D",
      "why": {
        "A": "More pages don't help when the right page is already retrieved, and may add distracting context.",
        "B": "Retrieval quality isn't the problem here.",
        "C": "Retrieval is already working. Changing the retriever doesn't fix a generator that ignores its evidence.",
        "D": "Correct. This is 'hallucination despite retrieval', a generator-side failure. Grounded prompting reduces it, though it doesn't fully eliminate it."
      },
      "takeaway": "Retrieval and generation fail independently. Right page + wrong answer → fix the generation side (grounded prompting, verification)."
    },
    {
      "id": "q7",
      "type": "Concept",
      "stem": "What is the main reason vector databases use approximate nearest neighbour (ANN) indexes such as FAISS or HNSW?",
      "options": {
        "A": "Exact search doesn't scale to millions of vectors; ANN gives a large speed-up for a small loss in accuracy.",
        "B": "They let keyword and semantic scores be combined.",
        "C": "They are more accurate than exact search.",
        "D": "They remove the need to store document embeddings."
      },
      "correct": "A",
      "why": {
        "A": "Correct. ANN trades a little accuracy for big speed gains at scale.",
        "B": "Combining BM25 and dense scores is hybrid search, a separate idea.",
        "C": "ANN is slightly less accurate than exact search, not more.",
        "D": "An index still stores the embeddings. ANN organises them for fast lookup."
      },
      "takeaway": "ANN = fast, slightly approximate search. It exists because exact search is too slow at scale."
    },
    {
      "id": "q8",
      "type": "Compare",
      "stem": "A student asks how late interaction differs from simply cutting each page into many small chunks. Which answer is correct?",
      "options": {
        "A": "Small chunking produces more vectors per page than late interaction.",
        "B": "Late interaction uses OCR; small chunking doesn't.",
        "C": "They are the same thing with different names.",
        "D": "Small chunking changes the retrieval unit to smaller pieces; late interaction keeps the whole page as the retrieval unit but compares query tokens against many patch vectors inside it."
      },
      "correct": "D",
      "why": {
        "A": "Late interaction typically has many more vectors per page (about 1,000 patches).",
        "B": "Neither idea is defined by OCR, and ColPali's late interaction specifically avoids OCR.",
        "C": "They differ in what gets returned: small pieces vs whole pages.",
        "D": "Correct. This is the Q&A clarification. The page stays the unit (good for citing 'which page'), while matching happens at fine-grained level."
      },
      "takeaway": "Late interaction = page-level retrieval with fine-grained internal matching."
    },
    {
      "id": "q9",
      "type": "Concept",
      "stem": "A team wants to use a different embedding model for queries than the one ColPali used for pages. Both output vectors of the same length. Will this work?",
      "options": {
        "A": "Yes. Any two models with matching vector length can be compared.",
        "B": "Yes, as long as cosine similarity is used instead of dot product.",
        "C": "No. Unless the two models were trained or aligned into the same semantic space, the similarity scores are meaningless.",
        "D": "No, because ColPali only works with image queries."
      },
      "correct": "C",
      "why": {
        "A": "Matching dimension is necessary but not sufficient. The axes of two independently trained spaces mean different things.",
        "B": "Changing the similarity measure doesn't align two unrelated spaces.",
        "C": "Correct. ColPali uses the same model for queries and pages so they share one space.",
        "D": "ColPali queries are text. It's built for text queries over page images."
      },
      "takeaway": "Queries and documents must be embedded into the same trained space. Same vector length isn't enough."
    },
    {
      "id": "q10",
      "type": "Scenario",
      "stem": "A legal-tech startup's RAG system works well on general documents but retrieves poorly on court judgments. Their LLM answers fine when given the right passages. According to the course, what should they fine-tune first?",
      "options": {
        "A": "The ANN index parameters only.",
        "B": "Nothing. Switching to multimodal RAG will fix it.",
        "C": "The embedding (or reranking) model used for retrieval.",
        "D": "The LLM generator, so it memorises the court judgments."
      },
      "correct": "C",
      "why": {
        "A": "Index tuning affects speed and approximate-search accuracy, not whether the embeddings understand legal language.",
        "B": "Court judgments are text. Multimodal RAG adds cost without addressing domain shift.",
        "C": "Correct. Retrieval is the weak part (domain shift). The course says to fine-tune the embedding or reranker when retrieval is poor; domain knowledge itself comes from RAG.",
        "D": "The generator already answers well when given the right passages, and memorising documents isn't how RAG supplies knowledge."
      },
      "takeaway": "Poor retrieval in a new domain → fine-tune the embedder or reranker, not the LLM."
    }
  ]
};

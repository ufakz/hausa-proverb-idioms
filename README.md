# Hausa Proverb / Idiom Evaluation

Evaluating whether small open-weight LLMs can **complete** proverbs and idioms—first in English as a high-resource control, then in **Hausa** as a low-resource, orally rich language where figurative expressions carry heavy communicative weight.

## Motivation

Current large language models perform well on many linguistic benchmarks. Their handling of metaphorical language such as idioms and proverbs has been studied in high- and mid-resource settings ([MAPS](https://aclanthology.org/2024.naacl-long.112/); [Khoshtab et al.](https://arxiv.org/abs/2410.16461)). [Kim et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1099/) introduce MIDAS and argue that idiom understanding in LLMs goes beyond memorization: models combine internal knowledge with contextual and reasoning cues.

This project tests that hypothesis on **Hausa**. Hausa is low-resource in typical LLM pretraining corpora, yet idioms and proverbs play a central role in everyday oral communication. Models that cannot understand or generate such expressions will struggle with adoption in Hausa-speaking contexts. English items serve as a control to separate general proverb-completion skill from language-specific gaps.

## Hypothesis

If MIDAS-style hybrid memorization + reasoning transfers, Qwen3.5 models should improve with scale on both English and Hausa completion. If cultural and data exposure dominate, English accuracy may scale while Hausa remains flat or improves only weakly—pointing to a blind spot poorly covered by standard benchmarks.

## Task

**Proverb / idiom completion.** Each item provides a partial expression; the model must produce the canonical continuation.

| Field    | Description                          |
|----------|--------------------------------------|
| `id`     | Unique item id (e.g. `i001`)         |
| `prompt` | Incomplete proverb / idiom           |
| `answer` | Expected completion                  |

Example:

```json
{
  "id": "i001",
  "prompt": "A stitch in time....",
  "answer": "saves nine"
}
```

## Dataset

| File | Status | Role |
|------|--------|------|
| [`data/english.json`](data/english.json) | Placeholder (English) | High-resource control |
| `data/hausa.json` | To be added | Primary low-resource evaluation |

Hausa data will use the **same schema**.

## Models

We evaluate open-weight **Qwen3.5** models in the ~0.6–6B range ([Hugging Face filter](https://huggingface.co/models?num_parameters=min:0.6B,max:6B&sort=trending)). Qwen is among the strongest openly released model families at this scale: official model-card results report, for example, MMLU-Pro of **55.3** (2B) and **79.1** (4B), MMLU-Redux of **69.2** (2B) and **88.8** (4B), and multilingual MMMLU of **56.9** / **76.1** for the 2B and 4B models respectively ([0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B), [4B](https://huggingface.co/Qwen/Qwen3.5-4B)). That combination of competitive general and multilingual scores, open weights, and a clean size ladder makes Qwen3.5 a practical choice for this study.

We deliberately evaluate a **scaling ladder** (0.8B → 2B → 4B), rather than a single model, to test whether classical scaling trends hold for **low-resource figurative understanding**. Gains with size on English would indicate that proverb/idiom completion scales like many other capabilities; weaker or absent gains on Hausa would suggest that scale alone does not close the gap for underrepresented oral figurative language.

| Model | Parameters | Hub |
|-------|------------|-----|
| Qwen3.5-0.8B | 0.8B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Qwen3.5-2B | 2B | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Qwen3.5-4B | 4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |

### Frontier baselines (OpenRouter)

We also evaluate frontier proprietary models via [OpenRouter](https://openrouter.ai/) as an upper-bound reference against the open-weight ladder:

| Model | OpenRouter id |
|-------|---------------|
| GPT-5.6 | [`openai/gpt-5.6-sol`](https://openrouter.ai/openai/gpt-5.6-sol) |
| Claude Opus 5 | [`anthropic/claude-opus-5`](https://openrouter.ai/anthropic/claude-opus-5) |
| Gemini 3.5 | [`google/gemini-3.5-flash`](https://openrouter.ai/google/gemini-3.5-flash) |

Set `OPENROUTER_API_KEY` (env var or Colab secret) before running section 9 of the notebook.

## Metrics

- **Exact match (normalized):** lowercase; strip punctuation and ellipsis; compare prediction to gold `answer`.
- **Soft match:** gold answer appears as a substring of the normalized prediction (handles over-generation).

Primary report: accuracy by model size; secondary: English vs Hausa when both datasets are available.

## Repository layout

```
hausa-proverb-idioms/
  README.md
  data/
    english.json          # English placeholders
    hausa.json            # (add later)
  notebooks/
    eval_qwen35_proverbs.ipynb
  results/                # written by the notebook
```

## Running the experiment (Colab)

1. Open [`notebooks/eval_qwen35_proverbs.ipynb`](notebooks/eval_qwen35_proverbs.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Upload or clone this repo so `data/english.json` is reachable (or set `DATA_PATH` in the config cell).
3. Enable a GPU runtime (T4 is enough for 0.8B / 2B; 4B may need more RAM or `float16` / 4-bit loading). Frontier OpenRouter calls do not need a GPU.
4. Run sections 1–8 for the Qwen ladder. For frontier baselines, set `OPENROUTER_API_KEY` and run section 9.
5. Results are written under `results/` (local predictions, OpenRouter predictions, scaling/compare plots).

To evaluate Hausa later, point `DATA_PATH` at `data/hausa.json`.

## Related work

Liu et al. ([2024](https://aclanthology.org/2024.naacl-long.112/)) introduce MAPS, a multicultural dataset of proverbs and sayings, and show that multilingual LLMs often memorize expressions without reliably reasoning about them in conversational context. Khoshtab et al. ([2024](https://arxiv.org/abs/2410.16461)) compare LLM performance on multilingual idioms and similes, highlighting cross-lingual gaps and the effect of prompting strategies. Kim et al. ([2025](https://aclanthology.org/2025.emnlp-main.1099/)) release MIDAS, a large-scale idiom dataset across six languages, and argue that idiom understanding in LLMs arises from an interplay between memorized knowledge and reasoning—especially for compositional idioms.

This project extends that line of work to Hausa proverb/idiom completion with a small open-weight scaling ladder, using English as a control.

### Citations

```bibtex
@inproceedings{liu-etal-2024-multilingual,
    title = "Are Multilingual {LLM}s Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings",
    author = "Liu, Chen Cecilia and Koto, Fajri and Baldwin, Timothy and Gurevych, Iryna",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.112/",
    doi = "10.18653/v1/2024.naacl-long.112",
    pages = "2026--2046"
}

@article{khoshtab-etal-2024-comparative,
    title = "Comparative Study of Multilingual Idioms and Similes in Large Language Models",
    author = "Khoshtab, Paria and Namazifard, Danial and Masoudi, Mostafa and Akhgary, Ali and Sani, Samin Mahdizadeh and Yaghoobzadeh, Yadollah",
    journal = "arXiv preprint arXiv:2410.16461",
    year = "2024",
    url = "https://arxiv.org/abs/2410.16461"
}

@inproceedings{kim-etal-2025-memorization,
    title = "Memorization or Reasoning? Exploring the Idiom Understanding of {LLM}s",
    author = "Kim, Jisu and Shin, Youngwoo and Hwang, Uiji and Choi, Jihun and Xuan, Richeng and Kim, Taeuk",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1099/",
    doi = "10.18653/v1/2025.emnlp-main.1099",
    pages = "21678--21699"
}
```

## License

Research use. Dataset items you add (especially Hausa) should be attributed according to their sources.

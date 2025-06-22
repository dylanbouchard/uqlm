---
title: 'UQLM: A Python Package Enabling Uncertainty Quantification for Language Models'
tags:
  - Python
  - Large Language Model
  - Uncertainty Quantification
  - Hallucination
authors:
  - name: Dylan Bouchard
    orcid: 0009-0004-9233-2324
    affiliation: 1
  - name: Mohit Singh Chauhan
    orcid: 0000-0002-7817-0427
    affiliation: 1 
  - name: David Skarbrevik
    orcid: 0009-0005-0005-0408
    affiliation: 1
  - name: Ho-Kyeong Ra
    orcid: 0000-0002-2342-6296
    affiliation: 1
  - name: Viren Bajaj
    orcid: 0000-0002-9984-1293
    affiliation: 1
  - name: Zeya Ahmad
    orcid: 0009-0009-1478-2940
    affiliation: 1
affiliations:
 - name: CVS Health Corporation
   index: 1
date: 1 June 2025
bibliography: paper.bib

---

# Summary

Hallucinations, defined as instances where Large Language Models (LLMs) generate false or misleading content, pose a significant challenge that impacts the safety and trust of downstream applications. We introduce `uqlm`, a Python library enabling hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers that compute response-level confidence scores ranging from 0 to 1. We categorize these scorers into four distinct types: black-box UQ, which measures consistency by comparing multiple responses generated from the same prompt; white-box UQ, which derives uncertainty from token-level probabilities; LLM-as-a-Judge, where an LLM evaluates the correctness of a response; and ensembles, which combine other scorers through a tunable, weighted average. Ultimately, `uqlm` provides an off-the-shelf solution for UQ-based hallucination detection that can be easily integrated to enhance the reliability of LLM outputs.

# Statement of Need
Large language models (LLMs) have revolutionized the field of natural language processing, but their tendency to generate false or misleading content, known as hallucinations, significantly compromise safety and trust. LLM hallucinations are especially problematic because they often appear plausible, making them difficult to detect and posing serious risks in high-stakes domains such as healthcare, legal, and financial applications. As LLMs are increasingly deployed in real-world settings, monitoring and detecting hallucinations becomes crucial.


Traditional evaluation methods involve 'grading' LLM responses by comparing model output to human-authored ground-truth texts, an approach offered by toolkits such as Evals [@Openai] and G-Eval [@liu-etal-2023-g]. While effective during pre-deployment testing, these methods are limited in practice since users typically lack access to ground-truth data at generation time. This shortcoming motivates the need for generation-time hallucination detection methods. 


Existing solutions to this problem include source-comparison methods, internet-based grounding, and uncertainty quantification (UQ) methods. Toolkits that offer source-comparison scorers, such as RAGAS [@es2023ragasautomatedevaluationretrieval], Phoenix [@phoenix], DeepEval [@Ip_deepeval_2025], and others [@hu2024refcheckerreferencebasedfinegrainedhallucination; @uptrain; @zha-etal-2023-alignscore; @asai2023selfraglearningretrievegenerate] evaluate the consistency between generated content and input prompts. However, these methods can mistakenly validate responses that merely mimic prompt phrasing without ensuring factual accuracy. Toolkits that leverage Internet searches for fact-checking, such as FacTool [@chern2023factool], introduce delays and risk incorporating erroneous online information, failing to address the inherent uncertainty in model outputs. Lastly, although several UQ techniques have been proposed in the literature, their adoption in user-friendly, comprehensive toolkits remains limited. For example, while SelfCheckGPT [@manakul-etal-2023-selfcheckgpt] incorporates some UQ scorers, its set of techniques is narrow and does not integrate generation with evaluation, thus creating barriers for practitioners outside specialized AI research environments. LangKit [@langkit] and NeMo Guardrails [@rebedea-etal-2023-nemo] also offer UQ scorers but are similarly narrow in scope.

UQLM aims to bridge these gaps by offering a comprehensive open-source toolkit that democratizes advanced research in uncertainty quantification for language models. Our toolkit implements a diverse array of uncertainty estimation techniques to compute generation-time, response-level confidence scores and uniquely integrates generation and evaluation processes. This integrated approach allows users to generate and assess content simultaneously, without the need for ground-truth data or external knowledge sources, and with minimal engineering effort. This democratization of access empowers smaller teams, researchers, and developers to incorporate robust hallucination detection into their applications, contributing to the development of safer and more reliable AI systems. 


# Usage

The `uqlm` library provides a collection of uncertainty quantifiers spanning four categories: black-box UQ, white-box UQ, LLM-as-a-Judge, and ensembles. The corresponding classes for these techniques, listed in the table below, are instantiated by passing an LLM object to the constructor.^[For the current version of `uqlm`, a LangChain `BaseChatModel` is required. Note that an LLM is not required for black-box scoring if users provide pre-generated responses.] Each of these classes contain a `generate_and_score` method, which generates LLM responses to a user provided list of prompts and computes response-level confidence scores. These scores range from 0 to 1, with higher confidence scores indicating lower likelihood that a response contains a hallucination or error.



| Scorer Class     | Compatibility | Added Latency | Added Cost |
|------------------|---------------|---------------|------------|
| `BlackBoxUQ`     | Universal     | Medium-High   | High       |
| `WhiteBoxUQ`     | Limited       | Minimal       | None       |
| `LLMPanel`       | Universal     | Low-High      | Low-High   |
| `UQEnsemble`     | Flexible      | Flexible      | Flexible   |


## Black-Box Uncertainty Quantification
Black-box uncertainty quantification exploits the stochastic nature of LLMs and measures the consistency of multiple responses to the same prompt. These consistency measurements can be conducted with various approaches, including cosine similarity, semantic entropy [@Farquhar2024], non-contradiction probability [@chen-mueller-2024-quantifying; @lin2024generatingconfidenceuncertaintyquantification], BERTScore [@manakul-etal-2023-selfcheckgpt; @zha-etal-2023-alignscore], BLEURT [@sellam-etal-2020-bleurt], and exact match rate [@cole-etal-2023-selectively]. Black-box UQ scorers are compatible with any LLM, but increase latency and generation costs. The corresponding class for this collection of scorers is `BlackBoxUQ`.

![BlackBoxGraphic](../assets/images/black_box_graphic.png)


To implement `BlackBoxUQ.generate_and_score`, users provide a list of prompts. For each prompt, an original response, along with additional candidate responses, are generated by the user-provided LLM, and consistency scores are computed using the specified scorers. If users set `use_best=True`, the uncertainty-minimized response is selected.^[Uncertainty-minimized response selection is based on the semantic entropy approach [@Farquhar2024]]. Below is a minimal example illustrating usage of `BlackBoxUQ`.

```python
from uqlm import BlackBoxUQ
bbuq = BlackBoxUQ(llm=llm, scorers=["entropy_score"])
results = await bbuq.generate_and_score(prompts=prompts, num_responses=5, use_best=True)
```


## White-Box Uncertainty Quantification
White-box uncertainty quantification leverages token probabilities to compute uncertainty. These approaches have the advantage of using the token probabilities associated with the generated response, meaning they do not add any latency or generation cost. However, because token probabilities are not accessible from all APIs, white-box scorers may not be compatible with all LLM applications. This collection of scorers can be implemented with the `WhiteBoxUQ` class, which offers two scorers: minimum token probability [@manakul-etal-2023-selfcheckgpt] and length-normalized token probability [@malinin2021uncertaintyestimationautoregressivestructured]. Below is a minimal example of `WhiteBoxUQ` usage.

```python
from uqlm import WhiteBoxUQ
wbuq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])
results = await wbuq.generate_and_score(prompts=prompts)
```

![WhiteBoxGraphic](../assets/images/white_box_graphic.png)


## LLM-as-a-Judge
LLM-as-a-Judge uses an LLM to evaluate the correctness of a response to a particular question. To achieve this, a question-response concatenation is passed to an LLM along with instructions to score the response's correctness. For LLM-as-a-Judge implementation, `uqlm` offers two classes: `LLMJudge` and `LLMPanel`.

![JudgeGraphic](../assets/images/judges_graphic.png)

### `LLMJudge` class
The `LLMJudge` class provides an off-the-shelf solution to implement LLM-as-a-Judge with a single LLM. Three off-the-shelf templates can be used: correct/incorrect (0/1), correct/uncertain/incorrect (0/0.5/1), and continuous (any value from 0 to 1). To score responses, users provide a list of questions (`prompts`) and a list of responses (`responses`) to the `LLMJudge.judge_responses` method. The returned result contains response-level confidence scores.

### `LLMPanel` class
The `LLMPanel` class enables users to compute confidence scores using  multiple LLM judges simultaneously. In the constructor, users pass a list of LLMs or pre-constructed `LLMJudge` instances to the `judges` argument. Implementing the `generate_and_score` method returns the score from each judge and aggregations of these scores, including minimum, maximum, average, and median. See below for a minimal example and an illustration of this workflow.

```python
from uqlm import LLMPanel
panel = LLMPanel(llm=llm1, judges=[llm1, llm2, llm3])
results = await panel.generate_and_score(prompts=prompts)
```

## Ensemble Approach
Lastly, `uqlm` offers both tunable and off-the-shelf ensembles that leverage a weighted average of any combination of the aforementioned confidence scores. Users specify which scorers to include using the `scorers` parameter in the constructor. Below, we detail the scoring approach and subsequently outline the optional tuning step that users may conduct prior to scoring.

### Ensemble Scoring
Similar to the aforementioned classes, `UQEnsemble` enables simultaneous generation and scoring with a `generate_and_score` method. Using the specified scorers, the ensemble score is a weighted average of the individual confidence scores, where weights may be default weights, user-specified, or tuned (see the following section for details). The off-the-shelf implementation follows an ensemble of exact match, non-contradiction probability, and self-judge proposed by @chen-mueller-2024-quantifying.

![GenerateScore](../assets/images/uqensemble_generate_score.png)


### Ensemble Tuning
Users may opt to tune the ensemble weights using the `tune` method prior to using the `generate_and_score` method. To implement this approach, users must provide a list of prompts and corresponding ideal responses to serve as an 'answer key'. The LLM's responses to the prompts are graded with a grader function that compares against the provided ideal responses. If a grader function is not provided by the user, the default grader function that leverages `vectara/hallucination_evaluation_model` is used. 

Once the binary grades (correct/incorrect) are obtained, an optimization routine solves for the optimal weights according to a specified classification objective. The objective function may be threshold-agnostic, such as ROC-AUC, or threshold-dependent, such as F1-score. After completing the optimization routine, the optimized weights are stored as class attributes to be used for subsequent scoring. Below is a minimal example illustrating this process. 

![GenerateScore](../assets/images/uqensemble_tune.png)

```python
from uqlm import UQEnsemble
## ---Option 1: Off-the-Shelf Ensemble---
# uqe = UQEnsemble(llm=llm)
# results = await uqe.generate_and_score(prompts=prompts, num_responses=5)

## ---Option 2: Tuned Ensemble---
scorers = [ # specify which scorers to include
    "exact_match", "noncontradiction", # black-box scorers
    "min_probability", # white-box scorer
    llm # use same LLM as a judge
]
uqe = UQEnsemble(llm=llm, scorers=scorers)

# Tune on tuning prompts with provided ground truth answers
tune_results = await uqe.tune(
    prompts=tuning_prompts, ground_truth_answers=ground_truth_answers
)
# ensemble is now tuned - generate responses on new prompts
results = await uqe.generate_and_score(prompts=prompts)
results.to_df()
```

# Author Contributions
Dylan Bouchard was the principal developer and researcher of the UQLM project, responsible for conceptualization, methodology, and software development of the UQLM library. Mohit Singh Chauhan helped lead research and software development efforts. David Skarbrevik, Ho-Kyeong Ra, Viren Bajaj, and Zeya Ahmad contributed to software development. 

# Conflict of Interest
The authors are employed and receive stock and equity from CVS Health® Corporation.



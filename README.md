<h1 align="center"><b>SAI'S AI ENGINEER EXECUTION WORKBOOK</b></h1>

30-project, trainer-guided build path

*Concept → Aman build → Your extension → Evaluation → Interview defense → GitHub evidence*

|  |
| --- |
| **PURPOSE This workbook is the missing execution layer. Your earlier trainer track tells you what to learn; this document tells you exactly what to open, what to study first, what to build, what to change, and what evidence you must produce before moving on.** |

# How to use this workbook

* The provided 138-project catalogue is a knowledge map, not a 138-project to-do list. This workbook selects a focused 30-project execution route.
* For each project: study only the listed prerequisite material → open the Aman article → rebuild it yourself → complete the trainer extension → run the exit test → write a Diary entry → commit evidence to GitHub.
* Do not move forward because the code runs. Move forward when you can explain the system and defend the design choices.
* Your trainer is ChatGPT. Bring your Diary update, code/output, errors, metrics or architecture diagram; I will review, quiz and decide whether the checkpoint is passed.
* SKIM/OPTIONAL items from the larger catalogue are intentionally excluded here.

# Resource rule

The resource list is intentionally narrow. You are not expected to finish an entire course before starting the associated project. Study only the concepts listed under 'Before you build'; ask the trainer when a concept is unclear.

# Status legend

|  |  |
| --- | --- |
| **Status** | **Meaning** |
| **CORE** | Must complete properly; portfolio/interview critical. |
| **BUILD** | Build once to learn the skill; lighter polish is acceptable. |
| **Existing** | Your existing banking ML project; use it as the first checkpoint. |

# Official resource index

* [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
* [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)
* [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning)
* [Kaggle Explainable AI](https://www.kaggle.com/learn/machine-learning-explainability)
* [fast.ai Practical Deep Learning](https://course.fast.ai/)
* [Karpathy Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
* [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)
* [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
* [Google Introduction to Generative AI](https://www.cloudskillsboost.google/course_templates/536)
* [DeepLearning.AI Advanced Retrieval for AI](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)
* [LangChain docs](https://docs.langchain.com/)
* [LlamaIndex docs](https://docs.llamaindex.ai/)
* [FastAPI docs](https://fastapi.tiangolo.com/)
* [Docker Get Started](https://docs.docker.com/get-started/)
* [GitHub Actions Quickstart](https://docs.github.com/en/actions/get-started/quickstart)
* [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview)
* [CrewAI docs](https://docs.crewai.com/)
* [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
* [PyTorch tutorials](https://docs.pytorch.org/tutorials/)
* [MLflow docs](https://mlflow.org/docs/latest/)
* [Python Packaging User Guide](https://packaging.python.org/en/latest/)
* [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
* [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

# 30-project route at a glance

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **#** | **Phase** | **Project** | **Priority** | **Main evidence** |
| 1 | 0 | 7-Day Customer Re-contact Prediction | CORE | Baseline + leakage/metric analysis |
| 2 | 1 | House Rent Prediction | BUILD | Regression + residuals |
| 3 | 1 | Loan Approval Prediction | BUILD | Business threshold analysis |
| 4 | 1 | Compare Multiple Predictive Models | BUILD | Model comparison matrix |
| 5 | 1 | Classification on Imbalanced Data | CORE | Imbalance + PR-AUC/thresholds |
| 6 | 2 | End-to-End Predictive Model | CORE | Train/eval pipeline |
| 7 | 2 | Packaging Machine Learning Models | CORE | Package + tests |
| 8 | 2 | REST Prediction API | CORE | REST endpoint |
| 9 | 2 | Dockerize the API | CORE | Docker image |
| 10 | 2 | Experiment Tracking Lab | CORE | Tracked experiments |
| 11 | 3 | Predictive Keyboard with PyTorch | CORE | PyTorch training loop |
| 12 | 3 | Transformer Text Classification | CORE | Transformer vs TF-IDF |
| 13 | 4 | Local LLM Application | CORE | Local LLM + structured output |
| 14 | 4 | Semantic Search with Embeddings | CORE | Semantic search |
| 15 | 4 | Text-to-SQL | CORE | NL→SQL + validation |
| 16 | 4 | AI SQL Assistant | CORE | SQL assistant + guardrails |
| 17 | 5 | RAG From Scratch | CORE | RAG from scratch |
| 18 | 5 | Document Q&A | CORE | Document Q&A + citations |
| 19 | 5 | Local RAG | CORE | Local/API comparison |
| 20 | 5 | Multi-Document RAG | CORE | Multi-doc + metadata |
| 21 | 5 | RAG Evaluation | CORE | Automated evals |
| 22 | 6 | Web-Connected Agent | CORE | Agent logs |
| 23 | 6 | Multi-Tool Agent | CORE | 3+ tools + failure handling |
| 24 | 6 | Research Agent | BUILD | Research report |
| 25 | 6 | Agentic RAG | CORE | Fixed vs agentic RAG |
| 26 | 6 | AI Data Analyst | CORE | Flagship analyst agent |
| 27 | 7 | LangGraph Workflow / Multi-Agent System | CORE | State graph |
| 28 | 7 | MCP Server | CORE | MCP server |
| 29 | 7 | MCP + Agent | BUILD | MCP agent |
| 30 | 8 | Production AI Service | CORE | Production service + CI/CD |

# 0 — Your existing ML flagship

## Project 1 — 7-Day Customer Re-contact Prediction

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source Your existing project** | **Checkpoint Required** |

**Project resource(s):** Use your existing 7-Day Customer Re-contact Prediction project/blueprint. No external tutorial is required for the first pass.

### What you learn

Binary classification in a banking/customer-contact setting; leakage control; train/test discipline; Logistic Regression; tree ensembles; imbalance; ROC-AUC vs PR-AUC; threshold selection; SHAP; production scoring.

### Before you build

* [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — Read the classification + model evaluation sections only.
* [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) — Complete/refresh model validation, under/overfitting and random forests.

### Build sequence

Run your existing project from a clean kernel/environment. Do not improve it first. Establish the baseline: target, features, split strategy, models, metrics and current output.

### Trainer extension

Add a business-driven threshold analysis and one leakage test. Then add SHAP for the final model.

### Exit criterion

You can explain the target, split, leakage risks, metric choice, threshold and why the final model is appropriate without opening the notebook.

### Evidence to commit

GitHub repo + README + baseline notebook/script + metric table + confusion matrix + threshold plot + SHAP plot + one architecture diagram.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 1 — Classical ML

## Project 2 — House Rent Prediction

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority BUILD** | **Level Beginner** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [House Rent Prediction](https://amanxai.com/2022/08/15/house-rent-prediction-with-machine-learning/)

### What you learn

First regression pipeline: EDA, encoding, train/test split, RMSE/R² and baseline comparison.

### Before you build

* [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) — Model validation, underfitting/overfitting, random forests.
* [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — Linear regression and numerical evaluation.

### Build sequence

Follow Aman's project once. Rebuild it yourself rather than copying line-by-line.

### Trainer extension

Add a simple baseline and compare a linear model with a tree-based model. Inspect residuals.

### Exit criterion

Explain MAE vs RMSE vs R² and when a regression model is leaking information.

### Evidence to commit

Clean repo, reproducible environment, EDA, baseline, model comparison and residual analysis.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 3 — Loan Approval Prediction

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority BUILD** | **Level Beginner** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Loan Approval Prediction](https://amanxai.com/2023/05/15/loan-approval-prediction-using-python/)

### What you learn

Categorical preprocessing, classification workflow, Random Forest, confusion matrix and BFSI business framing.

### Before you build

* [Kaggle Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) — Model validation and random forests.
* [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) — Missing values and categorical variables.

### Build sequence

Rebuild Aman's project. Explicitly document which errors are more costly in a lending context.

### Trainer extension

Tune the decision threshold and show how precision/recall change.

### Exit criterion

Given a business cost for false positives and false negatives, you can justify a threshold.

### Evidence to commit

Model card, confusion matrix, threshold table and business interpretation.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 4 — Compare Multiple Predictive Models

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority BUILD** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Compare Multiple Predictive Models](https://amanxai.com/2024/02/19/compare-multiple-machine-learning-models/)

### What you learn

Consistent model comparison, baselines, cross-validation and metric-driven selection.

### Before you build

* [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html) — Review train/test split and cross-validation.
* [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) — Review classification metrics.

### Build sequence

Compare Logistic Regression, Decision Tree, Random Forest and a boosting model on one dataset.

### Trainer extension

Create one comparison table with mean CV score, holdout score, training time and inference time.

### Exit criterion

You can explain why a model is selected without saying only 'it had the highest accuracy'.

### Evidence to commit

Reusable comparison script + results table + short decision memo.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 5 — Classification on Imbalanced Data

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Classification on Imbalanced Data](https://amanxai.com/2024/04/01/classification-on-imbalanced-data-using-python/)

### What you learn

Resampling, class weights, threshold choice and precision/recall trade-offs.

### Before you build

* [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) — Imbalanced data and leakage sections.
* [Kaggle Explainable AI](https://www.kaggle.com/learn/machine-learning-explainability) — Model explainability basics.

### Build sequence

Rebuild the imbalance workflow and compare at least two imbalance strategies.

### Trainer extension

Evaluate PR-AUC, recall at a business precision target, and threshold sensitivity.

### Exit criterion

You can explain why accuracy can be misleading and when PR-AUC is more informative.

### Evidence to commit

Experiment matrix, PR curve, threshold table and failure-case analysis.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 2 — ML Engineering

## Project 6 — End-to-End Predictive Model

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [End-to-End Predictive Model](https://amanxai.com/2023/12/18/build-an-end-to-end-machine-learning-model/)

### What you learn

Moving from data/model/evaluation notebook to a usable application structure.

### Before you build

* [Python Packaging User Guide](https://packaging.python.org/en/latest/) — Read the basic project/package concepts.
* [MLflow docs](https://mlflow.org/docs/latest/) — Understand experiment/run/metric concepts.

### Build sequence

Rebuild Aman's end-to-end project with separate data, training, evaluation and inference code.

### Trainer extension

Add MLflow experiment tracking and a reproducible configuration.

### Exit criterion

You can train, save, load and score the model from a clean environment without relying on notebook state.

### Evidence to commit

src/ project, config, requirements, training command, evaluation command, MLflow runs and README.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 7 — Packaging Machine Learning Models

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Packaging Machine Learning Models](https://amanxai.com/2024/09/09/packaging-machine-learning-models-with-python/)

### What you learn

Python packaging, serialization and API-oriented structure.

### Before you build

* [Python Packaging User Guide](https://packaging.python.org/en/latest/) — Read project structure, pyproject.toml and dependency management.

### Build sequence

Turn the model into an installable/reusable Python package.

### Trainer extension

Add unit tests for preprocessing and prediction input validation.

### Exit criterion

A fresh environment can install the package and run a prediction example.

### Evidence to commit

Package + tests + pyproject.toml + sample input/output.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 8 — REST Prediction API

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Deploy Your First ML Model as a REST API](https://amanxai.com/2025/10/07/deploy-your-first-ml-model-as-a-rest-api/)

### What you learn

HTTP request/response model, prediction endpoint, schema validation and service boundaries.

### Before you build

* [FastAPI docs](https://fastapi.tiangolo.com/) — Read first steps and request-body/Pydantic sections.
* [Python Packaging User Guide](https://packaging.python.org/en/latest/) — Refresh dependency management.

### Build sequence

Expose the packaged model through a POST /predict endpoint.

### Trainer extension

Add input validation, health endpoint and structured error responses.

### Exit criterion

You can explain the request lifecycle from JSON input to model output.

### Evidence to commit

FastAPI service + tests + example curl/Python client + OpenAPI docs screenshot.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 9 — Dockerize the API

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Deploy a Machine Learning Model with Docker](https://amanxai.com/2025/09/23/deploy-a-machine-learning-model-with-docker/)

### What you learn

Container images, Dockerfile, reproducible runtime and service execution.

### Before you build

* [Docker Get Started](https://docs.docker.com/get-started/) — Complete the first container/build/run concepts.

### Build sequence

Containerize the REST API and run it locally.

### Trainer extension

Add a non-root user, healthcheck and environment-based configuration.

### Exit criterion

You can build the image, run the container and explain what belongs inside vs outside the image.

### Evidence to commit

Dockerfile + .dockerignore + run instructions + working container.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 10 — Experiment Tracking Lab

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source Your existing project** | **Checkpoint Required** |

**Project resource(s):** Use your existing 7-Day Customer Re-contact Prediction project/blueprint. No external tutorial is required for the first pass.

### What you learn

Experiment tracking, parameters, metrics, artifacts and reproducibility.

### Before you build

* [MLflow docs](https://mlflow.org/docs/latest/) — Read Tracking Quickstart.

### Build sequence

Take Project 6 and run at least three model experiments through MLflow.

### Trainer extension

Log dataset/version identifier, parameters, metrics, model artifact and a short run description.

### Exit criterion

You can reproduce the selected run and explain why it was chosen.

### Evidence to commit

MLflow experiment screenshot/export + README explaining selection.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 3 — Deep Learning + NLP

## Project 11 — Predictive Keyboard with PyTorch

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building a Predictive Keyboard Model with PyTorch](https://amanxai.com/2025/05/13/building-a-predictive-keyboard-model-with-pytorch/)

### What you learn

PyTorch tensors, datasets, model class, forward pass, loss, optimizer, backward pass and parameter updates.

### Before you build

* [Karpathy Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Watch the early neural-network/backprop lessons.
* [PyTorch tutorials](https://docs.pytorch.org/tutorials/) — Complete a basic tensor + training-loop tutorial.

### Build sequence

Rebuild the project and write the training loop yourself.

### Trainer extension

Change the model or sequence representation and compare validation loss.

### Exit criterion

You can write the core training loop from memory and explain what happens at forward/backward/update.

### Evidence to commit

PyTorch training script + loss curves + explanation of one batch through the model.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 12 — Transformer Text Classification

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Text Classification Pipeline with Hugging Face Transformers](https://amanxai.com/2025/07/22/text-classification-pipeline-with-hugging-face-transformers/)

### What you learn

Pretrained transformer pipeline, tokenization, inference and text classification.

### Before you build

* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — Study the transformer and tokenization chapters needed to understand the pipeline.
* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — Use the course as conceptual support; do not complete the whole course before building.

### Build sequence

Rebuild Aman's Hugging Face classification pipeline.

### Trainer extension

Compare the pretrained pipeline with a classical TF-IDF baseline.

### Exit criterion

You can explain tokenization, transformer embeddings and why the pretrained model can outperform TF-IDF on semantic language tasks.

### Evidence to commit

Baseline vs transformer comparison + examples of correct and incorrect predictions.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 4 — LLM Foundations

## Project 13 — Local LLM Application

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building Your First Local LLM App](https://amanxai.com/2026/01/13/building-your-first-local-llm-app/)

### What you learn

Local inference, model serving concept, prompts, generation parameters and basic app structure.

### Before you build

* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — Study introductory LLM/transformer concepts.
* [Google Introduction to Generative AI](https://www.cloudskillsboost.google/course_templates/536) — Complete the introductory generative-AI concepts.

### Build sequence

Run a local LLM application following Aman's guide.

### Trainer extension

Add structured JSON output and validation.

### Exit criterion

You can describe the lifecycle: user input → prompt/messages → model inference → output parsing.

### Evidence to commit

Local app + structured-output example + README.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 14 — Semantic Search with Embeddings

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building Your First Smart Search with Python](https://amanxai.com/2026/02/15/building-your-first-smart-search-with-python/)

### What you learn

Embeddings, vector representations, cosine similarity and semantic retrieval.

### Before you build

* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — Study embeddings/representation sections relevant to semantic search.

### Build sequence

Build the semantic search system.

### Trainer extension

Add top-k retrieval and compare semantic search with keyword matching on a small labelled set.

### Exit criterion

You can explain why embeddings allow semantically related text to be retrieved even without exact keyword overlap.

### Evidence to commit

Search demo + labelled query set + precision@k-style analysis.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 15 — Text-to-SQL

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build Your First Text-to-SQL App](https://amanxai.com/2026/02/08/build-your-first-text-to-sql-app/)

### What you learn

Natural language to SQL, schema context, validation and execution.

### Before you build

* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — LLM basics only.
* [Google Introduction to Generative AI](https://www.cloudskillsboost.google/course_templates/536) — Prompting and generative-AI fundamentals.

### Build sequence

Build the Text-to-SQL application against a small database.

### Trainer extension

Add SQL validation/allow-listing and reject destructive statements.

### Exit criterion

You can explain why generating SQL is not the same as safely executing SQL.

### Evidence to commit

NL→SQL demo + validation layer + 10 test questions + failure cases.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 16 — AI SQL Assistant

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Create an AI SQL Assistant with LangChain](https://amanxai.com/2026/05/13/create-an-ai-sql-assistant-with-langchain/)

### What you learn

Tool use around a database, LangChain orchestration and safe query handling.

### Before you build

* [LangChain docs](https://docs.langchain.com/) — Read tool calling and SQL/database integration sections relevant to the build.
* Project 15 — Must be completed first.

### Build sequence

Build Aman's SQL assistant.

### Trainer extension

Add query explanation, result summarization and guardrails for expensive or destructive queries.

### Exit criterion

You can diagram LLM → SQL tool → database → result → explanation and explain each trust boundary.

### Evidence to commit

SQL assistant + safety checks + test set + architecture diagram.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 5 — RAG

## Project 17 — RAG From Scratch

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build Your First RAG System From Scratch](https://amanxai.com/2025/10/21/build-your-first-rag-system-from-scratch/)

### What you learn

Ingestion → chunking → embeddings → retrieval → prompt → generation, without hiding the pipeline behind a framework.

### Before you build

* [DeepLearning.AI Advanced Retrieval for AI](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) — Use as retrieval background after understanding embeddings.
* [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — LLM basics.

### Build sequence

Implement the RAG loop manually.

### Trainer extension

Add a retrieval debug mode that shows chunks, similarity scores and source metadata.

### Exit criterion

You can draw the RAG pipeline and explain what each stage contributes.

### Evidence to commit

From-scratch RAG + retrieval debug output + 10 labelled questions.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 18 — Document Q&A

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building a Document Q&A System](https://amanxai.com/2026/04/08/building-a-document-qa-system/)

### What you learn

Document ingestion, chunking, retrieval and question answering over documents.

### Before you build

* Project 17 — Must understand the from-scratch pipeline.

### Build sequence

Build document Q&A using Aman's approach.

### Trainer extension

Add source citations in answers and a 'not found in documents' behavior.

### Exit criterion

The system can answer supported questions and explicitly fail on unsupported ones.

### Evidence to commit

PDF/document Q&A app + cited answers + unsupported-question tests.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 19 — Local RAG

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build a Local RAG System with Open-Source LLMs](https://amanxai.com/2026/03/11/build-a-local-rag-system-with-open-source-llms/)

### What you learn

Local embedding/retrieval and local LLM generation.

### Before you build

* Project 18 — Document Q&A fundamentals.
* Project 13 — Local LLM basics.

### Build sequence

Rebuild the local RAG system.

### Trainer extension

Compare local vs API-based generation on the same 10-question test set.

### Exit criterion

You can discuss the quality/latency/cost/privacy trade-offs of local inference.

### Evidence to commit

Local RAG + comparison table.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 20 — Multi-Document RAG

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building a Multi-Document RAG System](https://amanxai.com/2026/01/06/building-a-multi-document-rag-system/)

### What you learn

Multiple-source ingestion, metadata, filtering and retrieval orchestration.

### Before you build

* Project 18 — Document Q&A.
* Project 14 — Embeddings and semantic search.

### Build sequence

Build multi-document retrieval.

### Trainer extension

Add metadata filters and source-aware citations; experiment with chunk size/overlap.

### Exit criterion

You can explain why metadata filtering and chunking strategy affect retrieval quality.

### Evidence to commit

Multi-document corpus + metadata schema + retrieval test results.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 21 — RAG Evaluation

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build an Evaluation Pipeline for Your LLM App](https://amanxai.com/2026/05/06/build-an-evaluation-pipeline-for-your-llm-app/)

### What you learn

Test sets, retrieval evaluation, answer-quality checks, latency and failure-case tracking.

### Before you build

* [DeepLearning.AI Advanced Retrieval for AI](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) — Review retrieval quality concepts.
* Aman's Custom Evals article — Read the evaluation-design article after the first eval pipeline.

### Build sequence

Create a labelled evaluation set for your RAG app and an automated evaluation script.

### Trainer extension

Track retrieval quality, answer quality, latency and representative failure cases.

### Exit criterion

You can tell whether a RAG change improved the system using evidence rather than intuition.

### Evidence to commit

eval dataset + evaluation script + baseline metrics + failure taxonomy.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 6 — Agents

## Project 22 — Web-Connected Agent

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Beginner** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Connect Your First AI Agent to the Internet](https://amanxai.com/2026/02/01/connect-your-first-ai-agent-to-the-internet/)

### What you learn

Agent loop, tool selection, tool execution and observation.

### Before you build

* [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) — Unit 1: agent fundamentals, tools and Thought-Action-Observation.
* [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) — Lessons 01–06 for vocabulary and basic architecture.

### Build sequence

Build the web-connected agent.

### Trainer extension

Log every tool call and cap the maximum number of steps.

### Exit criterion

You can explain why this is an agent rather than a single LLM call.

### Evidence to commit

Agent + step logs + loop limit + failure handling.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 23 — Multi-Tool Agent

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build a Multi-Tool AI Agent](https://amanxai.com/2026/02/25/build-a-multi-tool-ai-agent/)

### What you learn

Tool selection, tool schemas, observation and iterative execution.

### Before you build

* Project 22 — Basic agent loop.
* [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) — Tools + Thought/Action/Observation.

### Build sequence

Build an agent with at least three tools.

### Trainer extension

Add tool-error handling and a final response that distinguishes tool evidence from model-generated text.

### Exit criterion

You can justify why each tool exists and show what happens when one fails.

### Evidence to commit

3+ tool agent + logs + tool error tests.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 24 — Research Agent

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority BUILD** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build an AI Agent to Automate Your Research](https://amanxai.com/2025/11/11/build-an-ai-agent-to-automate-your-research/)

### What you learn

Multi-step research, retrieval, synthesis and source-aware outputs.

### Before you build

* Project 23 — Multi-tool agent.
* [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) — Planning/tool lessons relevant to multi-step workflows.

### Build sequence

Rebuild the research workflow.

### Trainer extension

Require source collection and a structured research report.

### Exit criterion

You can diagram the research loop and identify where hallucinations or tool errors can occur.

### Evidence to commit

Research agent + source list + report template + failure cases.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 25 — Agentic RAG

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Building an Agentic RAG Pipeline](https://amanxai.com/2025/12/30/building-an-agentic-rag-pipeline/)

### What you learn

Agent decides when/what to retrieve; retrieval becomes a tool rather than a fixed pipeline.

### Before you build

* Project 21 — RAG evaluation.
* Project 23 — Multi-tool agent.

### Build sequence

Build the agentic RAG pipeline.

### Trainer extension

Compare fixed RAG vs agentic RAG on the same labelled test set.

### Exit criterion

You can explain when agentic retrieval is justified and when fixed RAG is simpler.

### Evidence to commit

Two architectures + shared evaluation set + comparison.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 26 — AI Data Analyst

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build Your Personal AI Data Analyst](https://amanxai.com/2025/11/25/build-your-personal-ai-data-analyst/)

### What you learn

LLM + SQL + Python + visualization + reasoning; natural-language analytics.

### Before you build

* Project 16 — AI SQL Assistant.
* Project 23 — Multi-tool agent.
* Project 21 — Evaluation.

### Build sequence

Build the data analyst agent.

### Trainer extension

Make SQL read-only, add Python analysis as a separate tool, and require evidence for numeric claims.

### Exit criterion

You can design the agent so that the model orchestrates analysis rather than inventing results.

### Evidence to commit

Flagship AI Data Analyst + architecture diagram + evaluation set + demo.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 7 — Advanced AI Engineering

## Project 27 — LangGraph Workflow / Multi-Agent System

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build a Multi-Agent System With LangGraph](https://amanxai.com/2025/12/09/build-a-multi-agent-system-with-langgraph/)

### What you learn

State, nodes, edges, conditional routing, loops and orchestration.

### Before you build

* [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview) — Read the core graph/state concepts.
* [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) — LangGraph framework unit.

### Build sequence

Rebuild the LangGraph project after understanding the underlying agent loop.

### Trainer extension

Add explicit state and a conditional branch; document why a graph is useful here.

### Exit criterion

You can draw the state graph and explain every node/edge.

### Evidence to commit

LangGraph app + state diagram + failure path.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 28 — MCP Server

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Intermediate** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build Your First MCP Server in Python](https://amanxai.com/2026/02/22/build-your-first-mcp-server-in-python/)

### What you learn

MCP server/client model, exposing tools, schemas and controlled access to capabilities.

### Before you build

* [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) — Study tool interfaces and agent frameworks first.
* Project 23 — Tool-calling foundations.

### Build sequence

Build the MCP server.

### Trainer extension

Expose one useful banking/data-analysis-safe tool and document its schema and permissions.

### Exit criterion

You can explain MCP as a protocol boundary rather than just another agent framework.

### Evidence to commit

MCP server + tool schema + client test + permissions notes.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

## Project 29 — MCP + Agent

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority BUILD** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [I Built an AI Agent with MCP and Python](https://amanxai.com/2026/08/26/i-built-an-ai-agent-with-mcp-and-python-heres-how/)

### What you learn

Connecting an agent to MCP tools and separating agent logic from tool implementation.

### Before you build

* Project 28 — MCP server.
* Project 23 — Multi-tool agent.

### Build sequence

Build the MCP-connected agent.

### Trainer extension

Use at least two MCP tools and add tool failure handling.

### Exit criterion

You can explain the boundary between the agent, MCP client and MCP server.

### Evidence to commit

Agent + MCP server + architecture diagram + tool logs.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# 8 — Production AI

## Project 30 — Production AI Service

|  |  |  |  |
| --- | --- | --- | --- |
| **Priority CORE** | **Level Advanced** | **Source AmanXAI** | **Checkpoint Required** |

**Project resource(s):** [Build a Production-Ready LLM API](https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/) • [How to Dockerize an AI Agent](https://amanxai.com/2026/04/04/how-to-dockerize-an-ai-agent/) • [Setting Up a CI/CD Pipeline for LLM Applications](https://amanxai.com/2026/09/27/setting-up-a-ci-cd-pipeline-for-llm-applications/)

### What you learn

API serving, reliability, Docker, automated tests/evals and CI/CD.

### Before you build

* [FastAPI docs](https://fastapi.tiangolo.com/) — Review service structure and request validation.
* [Docker Get Started](https://docs.docker.com/get-started/) — Review image/build/run concepts.
* [GitHub Actions Quickstart](https://docs.github.com/en/actions/get-started/quickstart) — Read workflow basics.

### Build sequence

Combine the three Aman production articles into one production-style service.

### Trainer extension

Use your AI Data Analyst or RAG/agent project as the service. CI must run unit tests and a small evaluation set on every commit.

### Exit criterion

You can explain deployment architecture, test strategy, eval strategy, failure handling, latency and cost logging.

### Evidence to commit

FastAPI service + Docker image + GitHub Actions workflow + tests + evals + cost/latency log + README.

### Diary prompt

Write: What I did / What I understand / What I cannot explain / Mistake or misconception / Evidence / Question for trainer.

# Trainer checkpoints

# Trainer checkpoints

## Checkpoint 1 — ML

Explain Logistic Regression, Decision Tree, Random Forest and boosting; choose precision/recall/F1/ROC-AUC/PR-AUC for a stated business problem; identify leakage.

## Checkpoint 2 — ML Engineering

Take a saved model → load it → predict → serve it through an API → run it in Docker.

## Checkpoint 3 — LLM

Explain tokens, embeddings, transformer concept at a high level, structured output and the request lifecycle.

## Checkpoint 4 — RAG

Draw: documents → chunks → embeddings → vector store → retrieval → prompt → LLM → answer. Explain every box.

## Checkpoint 5 — Agents

Demonstrate LLM → tool → observation → next action and explain why a plain LLM call is not automatically an agent.

## Checkpoint 6 — AI Engineer

Design an AI Data Analyst/enterprise decision-intelligence system with SQL, RAG and Python tools, evaluation, API, Docker and monitoring concepts without copying a template.

# Definition of done — every CORE project

* Clean run from a fresh kernel/environment.
* Problem statement and input/output contract.
* GitHub repository with README and reproducible setup.
* At least one meaningful change beyond the tutorial.
* Tests appropriate to the project.
* Architecture diagram.
* Metrics and failure cases where applicable.
* You can explain the main design decisions without the tutorial open.
* A Diary entry is submitted to the trainer.

# What NOT to do

* Do not restart Python/data basics unless a real gap appears.
* Do not build every regression/classification tutorial in the 138-project catalogue.
* Do not jump to LangGraph, CrewAI or MCP before basic tool calling and RAG are solid.
* Do not fine-tune just because fine-tuning appears on a roadmap.
* Do not call a Streamlit UI 'deployment'.
* Do not copy a tutorial line by line and count it as a completed portfolio project.
* Do not change the roadmap every week because you found another AI course.
* Do not optimize model accuracy before establishing a baseline and evaluation protocol.

# Weekly trainer workflow

1. Start by telling me which project you are on.
2. Send a Diary entry after each meaningful milestone.
3. Bring errors, outputs and confusion rather than silently fixing everything with another tutorial.
4. I will quiz you before allowing a checkpoint to pass.
5. After a checkpoint, I will assign the next project or a targeted remedial exercise if a gap is exposed.

# Source and scope note

This workbook is derived from the provided 'Aman Kharwal Project Curriculum — Revised Edition' and the 30-project trainer route established in our conversation. The source curriculum states that its 138-project ordering, tags, prerequisites and checkpoints are curriculum design rather than Aman's own ordering, and that not every project should be built. The AmanXAI links here were extracted from that provided curriculum. Companion resources are official/free learning resources selected to support the same concepts; they are not additional projects you must complete.

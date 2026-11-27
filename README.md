# Inferential adaptation space in LLM

## Set up

`python -m pip install -r requirements.txt`

`hf download sbordt/OLMo-2-1B-1x --local-dir checkpoints/olmo2-wd01`

`hf download sbordt/OLMo-2-1B-1x-WD1 --local-dir checkpoints/olmo2-wd1`

## Research questions

| Research question                                                                                         | Investigation                                              |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **RQ1**. What changes in parameter space as a consequence of pretraining weight decay?                    | [Checkpoint analysis](huggingface.co/collections/th135/pt) |
| **RQ2**. How do these differences influence the magnitude and direction of subsequent functional updates? | Mathematical formulation and controlled interventions      |
| **RQ3**. Can these properties predict downstream plasticity and retention?                                | Fine-tuning and evaluation                                 |

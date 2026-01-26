# NumPert: Numerical Perturbations to Probe Language Models for Veracity Prediction ([Link to the paper](https://arxiv.org/abs/2511.09971))

## Overview
Numpert aims to investigate the numerical reasoning and classification capabilities of large language models in claim-and-evidence-based text inputs. The primary goal is to develop more robust approaches for numerical reasoning in language models.

## Repository Structure

### code
..\code\data_preprocessing\raw_dataset_preprocessing for cleaning dataset, e.g. removing verdict fact-checkers verdict in evidence, createing binary dataset and tag numerical values with spaCy

..\code\llm_eval\ scripts to evaluate LLMs using different wrappers connecting to APIs.

..\code\metrics\ scripts to calculate scores

..\code\error_analysis\ scripts used to extract reasoning tokens from reasoning models for manual analysis


### data

Contains raw and perturbed data

### Results

Baseline (unperturbed claims) in a separate folder for the evaluated models. 

Subdirs for each model for claims that are perturbed. These subdirs are separated into zero shot and two shot evaluation, and lastly for some select models, the PAP (called neg_shot in the folder structure).



## Perturb steps:
Before perturbing and evaluating the models, we do perform some data proprocessing. Files are found in . ..\code\data_preprocessing\perturbutations\
  1. Remove the third class (Conflicting) class, so the dataset only contain True and False classifications with the create_binary_dataset.py
  2. remove_reference.py removes the last part of the evidence of the evidence document of the dataset, so the verdict is not concluded in the document–makes it so the model have to infer the correct values instead of relying on fact-checker verdict.
  3. process_claims.py normalizes the data from step 2. Converts words to numbers for claims, and does named entity recognication on claims to extract tokens with numerical values. 
  4. main_perturb.py to run all perturbation types on data from step 3.


## Evaluation steps: 
Use files from ..\code\llm_eval\ dir. We use different files for different models types (OpenAI, Google or open-weight models using Ollama wrapper). Use argsparser to configure models type, input/output paths, api-configs, and other miscellaneous configurations. 

Directory also includes a json_to_jsonl.py to set a response format indended for OpenAI's batch evaluation.



## Results

The following tables presents accuracy performance for different datasplits. Red -x indicates a drop; Green +x indicates an increase. Values in bold denote the highest accuracy within each perturbation setting, separated by open-weight and proprietary models. PAP denote the perturbation aware prompt setting.

### True → True evaluation

<img width="425" height="720" alt="Screenshot 2025-12-19 at 20 38 07" src="https://github.com/user-attachments/assets/f98e227e-3003-4cd4-88ba-e0228ae3d72e" />

### True → False evaluation
<img width="900" height="767" alt="Screenshot 2025-12-19 at 20 38 19" src="https://github.com/user-attachments/assets/8fb2a343-0ac7-485d-8c8a-0158c4174b2d" />

### False → False evaluation
<img width="474" height="756" alt="Screenshot 2025-12-19 at 20 38 37" src="https://github.com/user-attachments/assets/dc4daa48-92d9-4ff9-8492-d14a1c147e43" />


### False → False evaluation (exaggerated numbers)
<img width="737" height="756" alt="Screenshot 2025-12-19 at 20 38 49" src="https://github.com/user-attachments/assets/c7bc824a-7b0a-4b55-b003-65be204501e7" />



#### Arxiv citation:

```
@misc{aarnes2025numpertnumericalperturbationsprobe,
      title={NumPert: Numerical Perturbations to Probe Language Models for Veracity Prediction}, 
      author={Peter Røysland Aarnes and Vinay Setty},
      year={2025},
      eprint={2511.09971},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.09971}, 
}
```



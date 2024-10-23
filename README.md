

# **MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated With Knowledge Guidance**

  This repository contains the accompanying code & resources for the paper:

**MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated With Knowledge Guidance*** [Anonymous Authors] *ICLR 2025* [[pdf]](https://anonymous.url).

  

 MDBench introduces a new multi-document reasoning benchmark synthetically generated through knowledge-guided prompting



## Codebase Overview

  **data/** - contains the MDBench dataset 

  **data_generation_configs/** - contains prompts used within dataset generation process.  

  **model_configs/** - configuration details for all models used during evaluation.  

  **multi_agent_exploration/** - contains prompts used to automatically + manually spawn multi-agent configurations used as multi-agentic 
  baselines. 
  
  **scripts/bash/** - scripts used for initiating dataset generation + evaluation.  

  **scripts/dataset_generation/** - scripts + logic used during the data generation proces.  

  **slurm/** - scripts used to spawn jobs on job spooler (TODO: this is somewhat UM-specific).  

  
  
 
## Updates
**October 2024:** MDBench under review at ICLR 2025


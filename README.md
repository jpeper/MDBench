

# **MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated With Knowledge Guidance**

  This repository contains the accompanying code & resources for the paper:

**MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated With Knowledge Guidance** -- Joseph J. Peper, Wenzhao Qiu, Ali Payani, Lu Wang -- *Findings of ACL 2025* [[pdf]](https://arxiv.org/pdf/2506.14927).

  

 MDBench introduces a new multi-document reasoning benchmark synthetically generated through knowledge-guided prompting.

<img width="1364" height="952" alt="image" src="https://github.com/user-attachments/assets/32d789f9-147e-4af8-9802-e02c9e9bd93e" />


## Codebase Overview

  **data/** - contains the MDBench dataset -- **July 2025 Update: Dataset Will Be Released Soon To Huggingface: https://huggingface.co/launch/mdbench** 

  **data_generation_configs/** - contains prompts used within dataset generation process.  

  **model_configs/** - configuration details for all models used during evaluation.  

  **multi_agent_exploration/** - contains prompts used to automatically + manually spawn prototype multi-agent configurations used as multi-agentic 
  baselines. 
  
  **scripts/bash/** - scripts used for initiating dataset generation + evaluation.  

  **scripts/dataset_generation/** - scripts + logic used during the data generation proces.  

  **slurm/** - scripts used to spawn jobs on job spooler (TODO: this is somewhat UM-specific).  

  
  
 
## Updates
**July 2025** MDBench Accepted to ACL 2025 Findings  
**February 2025** MDBench under review at ARR  
**October 2024:** MDBench under review at ICLR 2025  


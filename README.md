# 📂 Project Overview

This repository contains four main modules, each with its own dedicated README:

- **Fine-Tune Module** (`fine_tune/`)  
  Fine-tuning the `facebook/bart-large-cnn` model with LoRA adapters.  
  ⇨ See [`src/fine_tune/README.md`](fine_tune/README.md)

- **Evaluation Helper** (`summarisation_eval/`)  
  Tools for evaluating abstractive summarisation models (ROUGE, BERTScore).  
  ⇨ See [`src/summarisation_eval/README.md`](evaluation/README.md)

- **Dataset Creation Module** (`dataset_creation/`)  
  Harvest and preprocess news articles into JSON-Lines datasets.  
  ⇨ See [`src/dataset_creation/README.md`](dataset_creation/README.md)

- **Streamlit App** (`ChatBotStreamlit/`)  
  Dockerized Streamlit web interface for interacting with the summarisation pipeline.  
  ⇨ See [`src/ChatBotStreamlit/README.md`](streamlit_app/README.md)

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. Navigate to the module you need:
   ```bash
   cd fine_tune
   # or
   cd evaluation
   # or
   cd dataset_creation
   # or
   cd streamlit_app
   ```
3. Follow the instructions in the module’s README.

---
**Due to time constraints there are improvments to be done**: 
- Logging modules (to log every step), try/except etc. 
- No configuration files and parsing arguments for the scripts.  
- No clear structure between files and functions and each module. As in the code 
are 4 different modules:  
  1. Create dataset  
  2.  fine tune 
  3. evaluation on dataset  
  4. inference and UI 


As a result, we might find duplicate code in some parts of the code base. 
In addition, while fine tuning, we didn’t consider the chunking. So for a long article, we 
truncate it.s

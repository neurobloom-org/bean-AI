# Agent3: Intent Classifier SLM

This repository contains the source code, training logic, and deployment configurations for a fine-tuned **Llama 3.2 1B** Small Language Model (SLM) specialized in 6-way intent classification.

## 🧠 The "Brain" (Model Weights)
Due to file size constraints, the quantized model weights are hosted on the Hugging Face Hub.
* **Model Link:** [pieterszharsh/llama-3.2-1b-intent-classifier](https://huggingface.co/pieterszharsh/llama-3.2-1b-intent-classifier)

## 🎯 Classification Categories
Our agent classifies all user input into one of these 6 intents:
1. **Therapeutic** | 2. **Casual** | 3. **Critical** | 4. **Task** | 5. **Game** | 6. **Music**

## 🛠️ Quick Start (Deployment)
We use **Ollama** for sub-second inference.

1. **Install Dependencies:**
   `pip install -r requirements.txt`

2. **Pull and Run Model:**
   `ollama run hf.co/pieterszharsh/llama-3.2-1b-intent-classifier`

3. **Run Inference Script:**
   `python agent3_classifier.py`

## 🧪 Training Methodology
The model was fine-tuned using **Unsloth** and **LoRA** on a custom synthetic dataset of ~600 samples. The full training notebook and dataset are available in the `/training` directory.
Engineering Project: End-to-End Deep Learning Systems
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

### **1. Executive Summary & Project Mandate**

**Mandate:** As a Machine Learning Engineer at a forward-thinking tech firm, you are tasked with developing, evaluating, and documenting a proof-of-concept (PoC) deep learning model. This PoC will serve as the baseline for a future production system. Your primary goal is to deliver a reproducible, well-documented project that rigorously evaluates a model's performance on a given task.

**Core Competencies:** This project will assess your ability in data management, model development, training and optimization, performance analysis, and technical communication.

---

### **2. Core Learning Objectives**

Upon successful completion of this project, you will have demonstrated proficiency in:

-   **System Design:** Selecting appropriate models, loss functions, and metrics for a given problem domain.
-   **GPU-based Implementation:** Building and training a neural network efficiently using PyTorch on GPU hardware.
-   **Engineering Rigor:** Ensuring reproducibility through configuration management, code structure, and random seeding.
-   **Critical Evaluation:** Analyzing model performance not just by the final metric, but through learning curves, error analysis, and ablation studies.
-   **Technical Communication:** Summarizing your methodology, results, and limitations in a concise engineering report integrated with live code.

---

### **3. Project Tracks (Choose One)**

Select **one** of the following engineering tracks. Each track description below details the datasets, required metrics, and relevant course chapters.

-   #### **Track A: Computer Vision - Image Classification**
    -   **Business Case:** Develop a model for automated quality control or retail product categorization.
    -   **Datasets:** **CIFAR-10** (primary) or **Oxford-IIIT Pet** (secondary).
    -   **Metrics:** Accuracy, Macro-F1 Score, Confusion Matrix.
    -   **Course Chapters:** 1–2–3–4–5–6–7–10.

-   #### **Track B: Computer Vision - Object Detection**
    -   **Business Case:** Prototype a perception system for an autonomous vehicle or an inventory management drone.
    -   **Datasets:** **coco128** (primary) or **Penn-Fudan Pedestrian** (secondary).
    -   **Metrics:** mAP@0.5, mAP@0.5:0.95, Precision/Recall curves.
    -   **Course Chapters:** 1–3–4–6–7–8–10.

-   #### **Track C: NLP - Text Classification**
    -   **Business Case:** Build a system to automatically tag news articles by topic for an online publisher.
    -   **Datasets:** **AG News** (primary) or **IMDb** (secondary, subsampled).
    -   **Metrics:** Accuracy, Macro-F1 Score.
    -   **Course Chapters:** 1–2–3–9–10.

-   #### **Track D: Tabular/Time-Series - Regression**
    -   **Business Case:** Create a model to predict housing valuations or forecast climate variables for energy grid balancing.
    -   **Datasets:** **California Housing** (primary) or **Jena Climate** (secondary, sliced).
    -   **Metrics:** MSE, MAE, R², Plot of predictions vs. ground truth.
    -   **Course Chapters:** 1–2–3–6–10.

---

### **4. Project Lifecycle & Milestones**

Follow this structured lifecycle to ensure success. Your implementation should be documented within your final notebook.

-   **M1: Problem Scoping & Data Validation.**
    -   Define your input, output, and evaluation metric.
    -   Author a **Data Card**: document its source, licensing, features, and potential biases in a Markdown cell.
    -   Implement robust train/validation/test splits.

-   **M2: Baseline Model Implementation.**
    -   Implement the specified baseline model (e.g., ResNet-18, YOLOv8n, LSTM).
    -   **Crucially, ensure one full batch of data can pass through your model without errors.**
    -   Run an initial training for a few epochs to establish a performance baseline.

-   **M3: Optimization & Regularization.**
    -   Implement a complete training and validation loop.
    -   Integrate **at least one regularization technique** (e.g., Weight Decay, Dropout) and **one learning rate scheduling** strategy (e.g., Cosine Annealing, StepLR).
    -   Implement **early stopping** to prevent overfitting and save the best model checkpoint (`best.pt`).

-   **M4: Ablation Studies & Analysis.**
    -   Systematically conduct **at least two experiments** by changing one hyperparameter at a time (e.g., learning rate, augmentation intensity, model depth).
    -   Document the results of each experiment in a clear table (a pandas DataFrame displayed in the notebook is perfect).
    -   Analyze your model's failures: inspect the confusion matrix or plot prediction errors. Discuss what the model gets wrong and hypothesize why.

-   **M5: Reporting & Final Delivery.**
    -   Clean, comment, and structure your code.
    -   Organize your final notebook to serve as the project's complete report, integrating text, code, and outputs.


---

### **5. Deliverables & Submission**

Your submission will consist of a **single, polished notebook** (`.ipynb` file) that is both a functional script and your final report.

**Submission Format:**
Submit your final `.ipynb` file or share a link to your Colab/Kaggle notebook with viewing permissions enabled. Ensure the notebook has been run from top to bottom so all outputs are visible.
> **Recommended:** A **private GitHub Repository** containing your full, runnable source code and configuration files. Save the link of this private GitHub Repository in a **Google Form (Link will be shared soon)**

Your final notebook must contain:

1.  **Self-Contained Code:** All code for data loading, model definition, training, and evaluation.
2.  **Saved Model Weights:** The code should save `best.pt`.
3.  **Embedded Engineering Report:** A narrative written in Markdown cells, structured into five sections (Problem, Methodology, Results, Analysis, Conclusion).
4.  **Integrated Metrics:** Display your final metrics and ablation tables directly in the notebook's output cells.

**Team.**
Pair.


**Due Date.**
30/11/2025 23:59 Paris time.

---

### **6. Submission Checklist**

Before submitting, ensure you can check off every item on this list. This is your final quality gate.

-   [ ] **Final Run:** I have restarted the kernel and run the entire notebook from top to bottom. All outputs are visible without errors.
-   [ ] **Reproducibility:** My random seeds are set. I have a cell listing the versions of key libraries (`torch`, `ultralytics`, etc.).
-   [ ] **Report Completeness:** All five sections of the Embedded Engineering Report are filled out.
-   [ ] **Code Quality:** My code is well-commented, and scratch code has been removed.
-   [ ] **Outputs Verified:** The `best.pt` file is saved. My metrics and plots are clearly displayed.
-   [ ] **Submission Format:** I am submitting a single `.ipynb` file that meets all requirements.

---

### **7. Evaluation Rubric**

-   **Code & Reproducibility (30%):** The notebook runs end-to-end. Code is clean and seeds are fixed.
-   **Methodology & Experimentation (40%):** The model and techniques are sound. Ablation studies provide clear insights.
-   **Integrated Report & Communication (30%):** The report narrative is professional, clear, and demonstrates critical thinking.

---

## **Appendix**

### **A) Environments & Setup**

Use these free, GPU-accelerated environments for your project.

*   **Google Colab**: Runtime -> Change runtime type -> T4 GPU.
*   **Kaggle Notebooks**: Offers free T4/P100 GPUs.
*   **Hugging Face Spaces (Optional)**: For deploying a small Gradio demo with your final model.

**Minimal Setup Cell:** Place this at the top of your notebook.

```python
# Install required libraries
!pip -q install torch torchvision torchmetrics==1.4.0 matplotlib tqdm ultralytics datasets scikit-learn

# Set seeds for reproducibility
import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### **B) Quickstart Baselines**

Use these snippets to get your baseline model running quickly.

**Track A: CV Classification (ResNet-18 Transfer Learning)**
```python
import torchvision as tv, torch.nn as nn, torch
train_tf = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor()])
val_tf   = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.ToTensor()])
trainset = tv.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10) # Adapt final layer to 10 classes
model = model.to(device)
```

**Track B: Object Detection (YOLOv8)**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load pretrained model
# First, validate inference works on a sample image
# model.predict('path/to/image.jpg', save=True)
# Then, fine-tune on the small dataset
model.train(data='coco128.yaml', epochs=5, imgsz=640)
```

**Track C: NLP Text Classification (Tokenizer)**
```python
from datasets import load_dataset
ds = load_dataset("ag_news")
# Toy tokenizer example to build a vocabulary
from collections import Counter; import re
def tok(t): return re.findall(r"\w+", t.lower())
vocab = {"<pad>":0,"<unk>":1}
for ex in ds["train"]["text"][:5000]: 
    for w in tok(ex): vocab.setdefault(w, len(vocab))
```

### **C) Troubleshooting Guide**

-   **Loss is NaN or not decreasing?** Your learning rate is likely too high. Lower it by a factor of 10 (e.g., `1e-3` -> `1e-4`). Verify your data and labels are correct on a single batch.
-   **Model is overfitting?** (High training accuracy, low validation accuracy). Add more regularization: increase `weight_decay`, add `Dropout` layers, or use stronger data augmentation (for CV).
-   **CUDA "Out of Memory" Error?** Reduce your `batch_size`. If that's not enough, reduce the image size (`imgsz`) or sequence length.
-   **NLP padding/shape errors?** Double-check that all sequences in a batch are padded to the same length. Use PyTorch's `pad_sequence` or a Hugging Face tokenizer with padding enabled.

### **D) Ethics & Data Use**

Use only permissively licensed datasets and cite your sources properly within the Data Card section of your report. Do not upload private or sensitive data to public notebooks.

In your report's final section, include a short paragraph discussing potential biases in your chosen dataset (e.g., demographic imbalance, limited scope) and the potential for misuse of the technology you are building.

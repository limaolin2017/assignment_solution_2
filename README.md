# **QLoRA Finetuning Report**

## **1. Introduction**  
This report summarizes the fine-tuning of **Qwen/Qwen2.5-3B-Instruct** using **QLoRA** for a **text classification** task. The goal was to train the model to classify customer inquiries (e.g., *employee_benefits*, *payroll*) while keeping GPU usage low. The training was done on an **NVIDIA L4** GPU.

---

## **2. Dataset**  

- **Total Records**: **2501** (from `available_conversations.csv`).  
- **Train/Validation Split**: ~2000 records.  
- **Test Set**: **500** records.  
- **Formats**:  
  - Training: `finetune_classification.json`  
  - Testing: `test_classification.json`  

---

## **3. Fine-tuning with QLoRA**  

### **3.1 Training Setup**  
The model was fine-tuned using **LitGPT** with the following command:  

```bash
litgpt finetune_lora Qwen/Qwen2.5-3B-Instruct \
  --precision bf16-true \
  --data JSON \
  --data.json_path finetune_classification.json \
  --data.val_split_fraction 0.1 \
  --out_dir out/custom-model
```

- **GPU**: NVIDIA L4 (~9GB usage)  
- **Time**: ~27 min  
- **Epochs**: ~5  
- **Validation Loss**: **0.062**  

---

## **4. Testing & Results**  

### **4.1 Testing Command**  
```bash
python test.py \
  --model_path out/custom-model/final \
  --test_json test_classification.json \
  --max_new_tokens 50
```

### **4.2 Performance**  
- **Accuracy**: **0.94**  
- **Weighted F1 Score**: **0.95**  
- **Key classes**: **90–99%** F1 score  
- **"Other" category**: **0.80** F1 score  

---

## **5. Conclusion**  
Fine-tuning **QLoRA + LitGPT** on an **NVIDIA L4** produced a **94% accuracy model**. Most categories performed well, though the **"other" category** was slightly weaker (**F1 = 0.80**).
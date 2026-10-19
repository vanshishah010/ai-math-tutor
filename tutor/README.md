# 🧑‍🏫 AI Step-by-Step Math Tutor

A hybrid symbolic and neural AI math tutor integrated into the Streamlit Math Solver application. It provides:
1. **Granular Step-by-Step Solutions** with explicit mathematical rules and explanations.
2. **Progressive Hint Ladder** (Level 1: Conceptual Strategy → Level 2: Next Algebraic Action → Level 3: Concrete Intermediate Step → Level 4: Full Solution).
3. **Interactive Step Verification ("Check My Step")** that tests intermediate steps using symbolic equivalence, diagnoses common student mistakes (sign errors, arithmetic slips, inverted operations), and gives targeted feedback without giving away the final answer.

---

## 🏛️ Hybrid Architecture

- **SymPy (Ground Truth Oracle)**:
  - Formulates symbolic expressions and equations.
  - Verifies equivalence of student steps via $\text{simplify}(a - b) == 0$ or equation solution-set equivalence ($\text{roots}(S_{\text{student}}) == \text{roots}(S_{\text{original}})$).
  - Classifies errors into specific algebraic categories (e.g. sign flip $\text{simplify}(a + b) == 0$, arithmetic error on constant terms, wrong operation).
- **Fine-Tuned Small LLM (Qwen2.5-Math-1.5B-Instruct)**:
  - Fine-tuned using 4-bit QLoRA on step-by-step reasoning datasets.
  - Generates conversational, encouraging, pedagogical explanations.
- **Rule-Based Fallback**:
  - If deep learning weights or GPU hardware are absent, the tutor seamlessly operates via rule-based SymPy step extraction templates.

---

## 📁 Package Directory Structure

```
tutor/
├── __init__.py                # Package exports (MathTutor, TutorSolution, StepFeedback, TutorHint)
├── core/
│   ├── __init__.py
│   ├── solver.py              # SymPy solver with granular step extraction & word problem templates
│   ├── verifier.py            # Step equivalence checker & diagnostic error classifier
│   ├── hints.py               # 4-tier progressive hint ladder engine
│   ├── explainer.py           # Lazy-loaded LLM explainer with rule-based fallback
│   └── tutor.py               # Main MathTutor coordinator class
├── data/
│   ├── __init__.py
│   ├── prepare_dataset.py     # Cleans, filters, and formats Kaggle 520K dataset into ChatML JSONL
│   └── processed/             # Output directory for train.jsonl, val.jsonl, test.jsonl
├── training/
│   ├── __init__.py
│   ├── config.yaml            # Hyperparameters for QLoRA fine-tuning
│   ├── finetune_lora.py       # Training script with SFTTrainer (Hugging Face transformers + peft + trl)
│   └── notebook_colab.ipynb   # 1-click executable Google Colab / Kaggle notebook
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py            # Accuracy benchmark using SymPy symbolic verification
├── tests/
│   ├── __init__.py
│   ├── test_solver.py         # Pytest tests for linear, quadratic, calculus, and simplification
│   ├── test_verifier.py       # Pytest tests for equivalence, sign error detection, wrong step feedback
│   └── test_hints.py          # Pytest tests for progressive hint levels
├── demo.py                    # Standalone CLI demo showing all 5 required problem categories
└── README.md                  # This documentation
```

---

## 🚀 Quickstart & API Reference

### 1. Initialize Tutor
```python
from tutor import MathTutor

tutor = MathTutor(default_variable="x", enable_llm=False, device="auto")
```

### 2. Solve Step-by-Step (`solve`)
```python
solution = tutor.solve("3*x + 7 = 22", operation="solve", variable="x")
print("Final Answer:", solution.final_answer)
for step in solution.steps:
    print(f"Step {step.step_number} [{step.title}]: {step.description}")
```

### 3. Request Progressive Hints (`hint`)
```python
# Levels 1 to 4:
hint_lvl1 = tutor.hint("3*x + 7 = 22", level=1)  # Conceptual strategy
hint_lvl2 = tutor.hint("3*x + 7 = 22", level=2)  # Next algebraic action
hint_lvl3 = tutor.hint("3*x + 7 = 22", level=3)  # Intermediate checkpoint
hint_lvl4 = tutor.hint("3*x + 7 = 22", level=4)  # Full solution reveal
print(hint_lvl1.hint_text)
```

### 4. Verify Student Intermediate Step (`check_step`)
```python
# Student submits a step:
feedback = tutor.check_step(
    problem="3*x + 7 = 22",
    previous_steps=[],
    student_step="3*x = 29",  # Student added 7 instead of subtracting 7
    variable="x"
)

print("Valid:", feedback.is_valid)       # False
print("Status:", feedback.status)         # 'arithmetic_error'
print("Feedback:", feedback.message)     # Targeted diagnostic message
print("Hint:", feedback.hint)             # Actionable guidance
```

---

## 🧪 Running Unit Tests

Run the full pytest suite (18 unit tests):
```bash
pytest tutor/tests
```

---

## 🎬 Running the Interactive Demo

The demo showcases all 5 required problem types (Linear equation, Quadratic equation, Derivative, Integral, Simplification):
```bash
python tutor/demo.py
```

---

## 📊 Dataset Preparation

Prepares the Kaggle 520K Math Reasoning dataset (or auto-detects local archive):
```bash
python tutor/data/prepare_dataset.py --max-samples 50000 --output-dir tutor/data/processed
```
- Filters out non-step responses to retain high-pedagogical quality.
- Deduplicates samples.
- Splits into 80% train, 10% validation, 10% test.
- Exports standard ChatML JSONL (`train.jsonl`, `val.jsonl`, `test.jsonl`).

---

## 🏋️ Fine-Tuning on Free GPU (Colab / Kaggle)

1. Open `tutor/training/notebook_colab.ipynb` in Google Colab (free T4 GPU).
2. Or run via CLI:
```bash
python tutor/training/finetune_lora.py --config tutor/training/config.yaml
```
Saves adapter weights to `models/qwen2.5-math-tutor-lora`.

---

## 📈 Evaluation Benchmark

Verify final answers and step equivalence against test problems:
```bash
python tutor/evaluation/evaluate.py --max-samples 50
```

---

## 🖥️ Running the Streamlit App

Run the integrated application:
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Your existing image upload, canvas drawing, and OCR remain fully functional, alongside the new **🧑‍🏫 AI Interactive Math Tutor** workspace!

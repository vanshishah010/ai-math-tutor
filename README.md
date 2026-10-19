# AI Math Solver & Step-by-Step Math Tutor

An offline-capable AI Math Solver and pedagogical Step-by-Step Math Tutor built with Streamlit and SymPy.

## What it does
- **Upload Equation Image**: Extracts mathematical equations locally using offline OCR without any external cloud API or API key.
- **Type Equations Directly**: Direct algebraic input with automatic symbol handling and sanitization.
- **Symbolic Solver**: Instant symbolic computation for equation solving, simplification, factoring, differentiation, and integration.
- **🧑‍🏫 Interactive Math Tutor**:
  - **Granular Step-by-Step Solutions**: Clear pedagogical explanations of every algebraic transformation.
  - **Progressive Hint Ladder**: Tiered hints (Level 1: Concept → Level 2: Next Action → Level 3: Intermediate Checkpoint → Level 4: Full Solution).
  - **Check My Step**: Submit your own step; the tutor verifies it symbolically, diagnoses mistakes (e.g. sign errors, arithmetic slips, inverted operations), and guides you without giving away the final answer.

## Setup
Recommended: Python 3.10–3.12.

1. Open a terminal in the project directory.
2. Create and activate a virtual environment:

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python -m streamlit run app.py
   ```
5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## 100% Offline & Private
Image recognition runs completely on your local machine using offline OCR (`winocr` on Windows or `pytesseract`). No API keys, no Google Gemini account, and no internet access required.

## Optional AI Explanation Mode (Fine-Tuned Qwen)
MathLab includes an optional neural explanation mode that augments SymPy derivations with pedagogical explanations from a fine-tuned `Qwen2.5-Math-1.5B-Instruct` LoRA adapter (`models/qwen2.5-math-tutor-lora`).

- **Default SymPy Mode**: Pure, ultra-fast symbolic engine running 100% locally on CPU without requiring GPU hardware or heavy neural dependencies (`requirements.txt`).
- **Running AI Mode**:
  1. Requires a CUDA-enabled GPU (NVIDIA GPU with compute capability $\ge 7.0$).
  2. Install training & neural dependencies:
     ```bash
     pip install -r requirements-train.txt
     ```
  3. Ensure the fine-tuned adapter is located at `models/qwen2.5-math-tutor-lora`.
  4. In the app sidebar, switch on **"Use AI explanation (fine-tuned Qwen)"**.
  - *Note*: If no CUDA GPU, neural libraries, or adapter directory are detected, the app automatically stays in SymPy engine mode with a clear notice. SymPy always remains the source of truth for all final answers and step verification.


## Input examples
- Solve: `3*x + 7 = 22` or `x^2 - 5*x + 6 = 0`
- Simplify: `(x^2 - 4)/(x - 2)`
- Factor: `x^2 - 5*x + 6`
- Differentiate: `x^3 + 2*x^2 - 5*x + 1`
- Integrate: `3*x^2 + 4*x - 2`

Use `*` for multiplication and `^` or `**` for powers.

## Running Tests & Demo
- Run the unit tests:
  ```bash
  pytest tutor/tests
  ```
- Run the interactive demo showing all 5 problem types:
  ```bash
  python tutor/demo.py
  ```

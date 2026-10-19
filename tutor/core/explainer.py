"""Neural and Rule-Based Pedagogical Explainer.

Wraps small language models (Qwen2.5-Math-1.5B-Instruct, fine-tuned LoRA adapters)
with:
1. Lazy loading (the model is never loaded into RAM/VRAM until explicitly requested).
2. Hardware awareness (CPU vs CUDA with automatic device mapping and bfloat16/float32 selection).
3. Robust rule-based fallback (if torch/transformers are missing or GPU/model weights are not present,
   it generates clean, template-based pedagogical explanations using SymPy without failing).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tutor.core.solver import Step, TutorSolution

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert, patient AI Math Tutor. Solve math problems step by step "
    "with clear explanations, pedagogical structure, and mathematical rigor. "
    "Never skip reasoning steps."
)

DEFAULT_ADAPTER_DIR = "models/qwen2.5-math-tutor-lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"


class RuleBasedExplainer:
    """Zero-dependency rule-based explainer fallback."""

    @staticmethod
    def explain(solution: TutorSolution) -> str:
        """Format solution steps into natural language explanations."""
        if not solution.success:
            return f"I encountered difficulty solving this problem symbolically: {solution.error_message}"

        lines = [f"### Step-by-Step Explanation for `{solution.problem}`\n"]
        for s in solution.steps:
            lines.append(f"**Step {s.step_number}: {s.title}**")
            lines.append(f"{s.description}")
            if s.latex:
                lines.append(f"$$\n{s.latex}\n$$")
            lines.append("")

        lines.append(f"**Final Answer:** `{solution.final_answer}`")
        return "\n".join(lines)


class LLMExplainer:
    """Lazy-loaded neural explainer with seamless rule-based fallback."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model_name_or_path = model_name_or_path or DEFAULT_BASE_MODEL
        self.adapter_path = adapter_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Lazy loading state
        self._model = None
        self._tokenizer = None
        self._load_attempted = False
        self._load_failed = False
        self._failure_reason: Optional[str] = None

    def is_available(self) -> bool:
        """Check whether neural libraries and model artifacts are available."""
        if self._load_failed:
            return False
        try:
            import torch
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> bool:
        """Perform lazy loading of tokenizer and model weights on first call."""
        if self._model is not None:
            return True
        if self._load_attempted and self._load_failed:
            return False

        self._load_attempted = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Determine device
            if self.device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = self.device.lower()

            logger.info("Lazy-loading LLM on device '%s'...", target_device)

            dtype = torch.bfloat16 if (target_device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if target_device == "cuda" else torch.float32)

            model_source = self.model_name_or_path

            self._tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=dtype,
                device_map=target_device if target_device != "cpu" else None,
                trust_remote_code=True,
            )

            # Check if adapter exists
            if self.adapter_path:
                adapter_dir = Path(self.adapter_path)
                if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
                    logger.info("Loading fine-tuned LoRA adapter from: %s", adapter_dir)
                    self._model = PeftModel.from_pretrained(base, str(adapter_dir))
                else:
                    logger.warning("Adapter path '%s' specified but adapter_config.json not found. Using base model.", self.adapter_path)
                    self._model = base
            else:
                logger.info("Using base model directly (no adapter specified)")
                self._model = base

            if target_device == "cpu":
                self._model = self._model.to("cpu")

            self._model.eval()
            logger.info("LLM successfully initialized.")
            return True

        except Exception as e:
            logger.warning("Could not load neural model: %s. Falling back to rule-based explainer.", e)
            self._load_failed = True
            self._failure_reason = str(e)
            return False

    def generate_explanation(self, problem: str, context_steps: Optional[List[str]] = None) -> str:
        """Generate pedagogical explanation text via LLM or rule-based fallback."""
        if not self._ensure_loaded():
            return f"Note: Running in rule-based mode ({self._failure_reason or 'Model not loaded'})."

        try:
            import torch

            user_content = f"Problem: {problem}"
            if context_steps:
                user_content += "\nGround-truth steps:\n" + "\n".join(f"- {s}" for s in context_steps)
                user_content += "\nPlease provide a clear, encouraging, step-by-step tutoring explanation."

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            return response

        except Exception as e:
            logger.warning("LLM text generation error: %s. Using rule-based fallback.", e)
            return f"Rule fallback: {e}"

    def enrich_solution(self, solution: TutorSolution) -> TutorSolution:
        """Augment a SymPy solution with LLM-generated explanations if available."""
        if not self.is_available() or not self._ensure_loaded():
            # Return solution as is with SymPy rule descriptions
            return solution

        step_descriptions = [f"{s.title}: {s.description}" for s in solution.steps]
        try:
            llm_text = self.generate_explanation(solution.problem, context_steps=step_descriptions)
            if llm_text:
                # Add an introductory overarching pedagogical guidance step
                solution.steps.insert(
                    0,
                    Step(
                        step_number=0,
                        title="Tutor Concept Overview",
                        description=llm_text[:300] + ("..." if len(llm_text) > 300 else ""),
                        expression="",
                        latex="",
                        rule_applied="Neural Tutor Overview",
                    )
                )
        except Exception as e:
            logger.debug("Enrichment skipped: %s", e)

        return solution

    def generate_solution(self, problem: str) -> str:
        """Prompt the LLM directly to solve a problem step-by-step."""
        if not self._ensure_loaded():
            return ""

        try:
            import torch

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            return response
        except Exception as e:
            logger.warning("LLM direct solve error: %s", e)
            return ""

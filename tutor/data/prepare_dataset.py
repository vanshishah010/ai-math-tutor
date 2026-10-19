"""Dataset preparation pipeline for fine-tuning small math LLMs (Qwen2.5-Math).

Loads dataset from local file/directory or downloads via kagglehub:
"Large-Scale Math Reasoning Dataset (520K)" by dineshkumar0705.
Cleans, deduplicates, filters for multi-step reasoning, splits into train/val/test,
and formats into ChatML instruction-tuning JSONL format.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYSTEM_PROMPT = (
    "You are an expert, patient AI Math Tutor. Solve math problems step by step "
    "with clear explanations, pedagogical structure, and mathematical rigor. "
    "Never skip reasoning steps."
)

DEFAULT_LOCAL_PATH = r"C:\Users\vanshi\Downloads\archive\datasetv2.jsonl\dataset.jsonl"
KAGGLE_DATASET_ID = "dineshkumar0705/large-scale-math-reasoning-dataset-520k"


class DatasetPreparer:
    """Prepares and formats math reasoning datasets for instruction tuning."""

    def __init__(
        self,
        input_path: Optional[str] = None,
        output_dir: str = "tutor/data/processed",
        max_samples: int = 50000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        random.seed(seed)

    def locate_dataset(self) -> Path:
        """Find the dataset file locally or download via kagglehub."""
        # 1. Explicit path given
        if self.input_path and Path(self.input_path).exists():
            p = Path(self.input_path)
            if p.is_dir():
                # Check for jsonl or csv inside
                for candidate in p.glob("*.jsonl"):
                    return candidate
                for candidate in p.glob("*.csv"):
                    return candidate
            return p

        # 2. Check known local archive path
        if Path(DEFAULT_LOCAL_PATH).exists():
            logger.info("Found local dataset archive at: %s", DEFAULT_LOCAL_PATH)
            return Path(DEFAULT_LOCAL_PATH)

        # 3. Check archive directory
        parent_dir = Path(r"C:\Users\vanshi\Downloads\archive\datasetv2.jsonl")
        if parent_dir.exists():
            for f in parent_dir.glob("*.jsonl"):
                return f

        # 4. Fallback to kagglehub download
        logger.info("Local file not found; downloading '%s' via kagglehub...", KAGGLE_DATASET_ID)
        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
            path_obj = Path(dataset_path)
            for f in path_obj.rglob("*.jsonl"):
                return f
            for f in path_obj.rglob("*.csv"):
                return f
            for f in path_obj.rglob("*.json"):
                return f
            raise FileNotFoundError(f"No suitable dataset file found in kagglehub path: {dataset_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset locally and via kagglehub: {e}\n"
                f"Please specify --input-path pointing to your dataset file."
            )

    @staticmethod
    def identify_columns(sample_record: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        """Automatically detect problem, response, and answer column keys."""
        keys = list(sample_record.keys())
        keys_lower = {k.lower(): k for k in keys}

        # Problem / Question candidates
        problem_candidates = ["prompt", "problem", "question", "input", "query", "instruction"]
        problem_key = None
        for cand in problem_candidates:
            if cand in keys_lower:
                problem_key = keys_lower[cand]
                break

        # Response / Solution candidates
        response_candidates = ["response", "solution", "rationale", "output", "reasoning", "explanation", "steps"]
        response_key = None
        for cand in response_candidates:
            if cand in keys_lower:
                response_key = keys_lower[cand]
                break

        # Answer candidates
        answer_candidates = ["answer", "final_answer", "target", "ground_truth"]
        answer_key = None
        for cand in answer_candidates:
            if cand in keys_lower:
                answer_key = keys_lower[cand]
                break

        if not problem_key or not response_key:
            raise ValueError(
                f"Could not auto-detect columns from schema keys: {keys}. "
                f"Detected problem={problem_key}, response={response_key}."
            )

        logger.info(
            "Auto-detected schema columns: problem='%s', response='%s', answer='%s'",
            problem_key, response_key, answer_key
        )
        return problem_key, response_key, answer_key

    def is_multi_step_solution(self, text: str) -> bool:
        """Filter out one-word or trivial non-step answers to ensure pedagogical quality."""
        if not text or len(text.strip()) < 40:
            return False

        t = text.lower()
        # Look for multi-step reasoning markers
        markers = [
            "step", "first", "second", "therefore", "hence", "because",
            "compute", "substitute", "using", "we get", "yields", "given",
            "goal", "equation", "simplify", "factor", "formula", "rewrite"
        ]
        marker_count = sum(1 for m in markers if m in t)
        # Accept if multiple sentences or multiple reasoning markers
        has_multiple_sentences = len(re.split(r"[.!?\n]", text.strip())) >= 2
        return marker_count >= 1 and has_multiple_sentences

    def read_records(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Stream raw records from JSONL or CSV."""
        if file_path.suffix.lower() == ".jsonl":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        elif file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item

    def process(self) -> Dict[str, int]:
        """Execute full cleaning, filtering, deduplication, splitting, and saving."""
        data_file = self.locate_dataset()
        logger.info("Reading dataset from: %s", data_file)

        # 1. Determine schema from first record
        first_record = next(self.read_records(data_file), None)
        if not first_record:
            raise ValueError(f"Dataset file is empty: {data_file}")
        prob_key, resp_key, ans_key = self.identify_columns(first_record)

        # 2. Stream, filter, and deduplicate
        seen_problems = set()
        cleaned_samples: List[Dict[str, Any]] = []

        total_scanned = 0
        for record in self.read_records(data_file):
            total_scanned += 1
            raw_prob = str(record.get(prob_key, "")).strip()
            raw_resp = str(record.get(resp_key, "")).strip()

            if not raw_prob or not raw_resp:
                continue

            # Check multi-step reasoning quality
            if not self.is_multi_step_solution(raw_resp):
                continue

            # Deduplication
            prob_hash = hash(raw_prob.lower().strip())
            if prob_hash in seen_problems:
                continue
            seen_problems.add(prob_hash)

            # Build ChatML record
            chat_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_prob},
                    {"role": "assistant", "content": raw_resp},
                ],
                "meta": {
                    "source_id": str(record.get("id", record.get("problem_id", total_scanned))),
                    "answer": str(record.get(ans_key, "")) if ans_key else "",
                },
            }
            cleaned_samples.append(chat_record)

            if len(cleaned_samples) >= self.max_samples:
                logger.info("Reached maximum requested sample limit of %d", self.max_samples)
                break

        logger.info(
            "Scanned %d raw rows -> Filtered & deduplicated %d high-quality step-by-step samples",
            total_scanned, len(cleaned_samples)
        )

        if not cleaned_samples:
            raise RuntimeError("No valid samples passed the multi-step reasoning filter!")

        # 3. Shuffle and split
        random.shuffle(cleaned_samples)
        n = len(cleaned_samples)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_data = cleaned_samples[:train_end]
        val_data = cleaned_samples[train_end:val_end]
        test_data = cleaned_samples[val_end:]

        # 4. Save to output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train": self.output_dir / "train.jsonl",
            "val": self.output_dir / "val.jsonl",
            "test": self.output_dir / "test.jsonl",
        }

        for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            with open(paths[name], "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("Saved %s split: %d samples to %s", name, len(split_data), paths[name])

        return {
            "total_processed": n,
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Math Reasoning Dataset for LLM Fine-Tuning")
    parser.add_argument("--input-path", type=str, default=None, help="Path to input jsonl/csv or archive")
    parser.add_argument("--output-dir", type=str, default="tutor/data/processed", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=50000, help="Subset size for free GPU training")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    preparer = DatasetPreparer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    results = preparer.process()
    print("Data preparation complete:", results)


if __name__ == "__main__":
    main()

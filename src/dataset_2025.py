"""Build a refreshed WikiMIA-style dataset with 32-word snippets from recent Wikipedia articles.

The resulting dataset mirrors the structure expected by ``run.py`` (``input`` text and
``label`` fields) but now defines ``label == 1`` when an article itself was *created*
before year 2025 (seen data) and ``label == 0`` when it was created during or after 2025
(unseen data). The script uses the
``wikipedia-api`` package when available and falls back to the MediaWiki API for content and
metadata retrieval.
By default it gathers roughly 500 snippets spread across random articles of mixed genres and
keeps the total roughly balanced between seen/unseen labels. Unseen (label-0) articles now
contribute up to fifty sequential 32-word snippets each so that multiple dataset rows may
originate from the same page, even if an article is too short to yield the full set of
chunks.

"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from datasets import Dataset

import wikipediaapi



MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "detect-pretrain-data-builder/1.0 (dataset generation script)"
CHUNKS_PER_LABEL_ZERO = 50
MIN_COMPLETION_RATIO = 0.9
BALANCE_TOLERANCE = 10


@dataclass
class ArticleSample:
	title: str
	input_text: str
	label: int
	created_at: str
	last_modified: str

	def to_dict(self) -> dict:
		return {
			"title": self.title,
			"input": self.input_text,
			"label": self.label,
			"created_at": self.created_at,
			"last_modified": self.last_modified,
		}


def _request_json(params: dict) -> dict:
	response = requests.get(MEDIAWIKI_API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
	response.raise_for_status()
	return response.json()


def fetch_random_titles(batch_size: int = 10) -> List[str]:
	params = {
		"action": "query",
		"format": "json",
		"list": "random",
		"rnnamespace": 0,
		"rnlimit": batch_size,
	}
	data = _request_json(params)
	print(f"Fetched {len(data.get('query', {}).get('random', []))} random titles.")
	print(f"First 5 titles: {[entry['title'] for entry in data.get('query', {}).get('random', [])[:5]]}")
	return [entry["title"] for entry in data.get("query", {}).get("random", [])]


def fetch_latest_timestamp(title: str) -> Optional[str]:
	params = {
		"action": "query",
		"prop": "revisions",
		"rvlimit": 1,
		"rvprop": "timestamp",
		"titles": title,
		"format": "json",
	}
	data = _request_json(params)
	pages = data.get("query", {}).get("pages", {})
	for page in pages.values():
		revisions = page.get("revisions")
		if revisions:
			return revisions[0].get("timestamp")
	return None


def fetch_creation_timestamp(title: str) -> Optional[str]:
	params = {
		"action": "query",
		"prop": "revisions",
		"rvlimit": 1,
		"rvdir": "newer",  # fetch the earliest revision
		"rvprop": "timestamp",
		"titles": title,
		"format": "json",
	}
	data = _request_json(params)
	pages = data.get("query", {}).get("pages", {})
	for page in pages.values():
		revisions = page.get("revisions")
		if revisions:
			return revisions[0].get("timestamp")
	return None


def fetch_text_with_wikipedia_api(title: str, min_words: int) -> Optional[str]:
	if wikipediaapi is None:
		return None
	wiki = wikipediaapi.Wikipedia(
		language="en",
		user_agent=USER_AGENT,
		extract_format=wikipediaapi.ExtractFormat.WIKI,
	)
	page = wiki.page(title)
	if not page.exists():
		return None
	# print(f"page has attributes: {dir(page)}")
	# print(f"page text: {page.text}")
	# print(f"sections: {[section.text for section in page.sections]}")
	description = page.text
	words = description.split()
	print(f"Fetched text for '{title}' with {len(words)} words using wikipedia api.")
	if len(words) < min_words:
		return None
	return " ".join(words)


def fetch_text_via_mediawiki(title: str, min_words: int) -> Optional[str]:
	params = {
		"action": "query",
		"prop": "extracts",
		"explaintext": 1,
		"exintro": 1,
		"titles": title,
		"format": "json",
	}
	data = _request_json(params)
	pages = data.get("query", {}).get("pages", {})
	for page in pages.values():
		extract = page.get("extract", "")
		words = extract.split()
		if len(words) >= min_words:
			print(f"Fetched text for '{title}' with {len(words)} words using mediawiki api.")
			return " ".join(words)
	return None


def build_samples(
	target_count: int,
	snippet_words: int,
	min_year: int = 2015,
	seed: int = 2025,
) -> List[ArticleSample]:
	random.seed(seed)
	collected: List[ArticleSample] = []
	seen_titles = set()
	attempts = 0
	max_attempts = target_count * 80
	desired_total = target_count
	min_total = max(1, int(desired_total * MIN_COMPLETION_RATIO))
	desired_label_zero = desired_total // 2
	desired_label_one = desired_total - desired_label_zero
	label_counts = {0: 0, 1: 0}

	def goals_met() -> bool:
		total = len(collected)
		balanced = abs(label_counts[0] - label_counts[1]) <= BALANCE_TOLERANCE
		return (total >= desired_total and balanced) or (
			label_counts[0] >= desired_label_zero and label_counts[1] >= desired_label_one
		)

	while attempts < max_attempts:
		attempts += 1
		titles = fetch_random_titles(batch_size=10)
		random.shuffle(titles)
		for title in titles:
			if title in seen_titles:
				continue
			seen_titles.add(title)

			text = fetch_text_with_wikipedia_api(title, snippet_words)
			if text is None:
				text = fetch_text_via_mediawiki(title, snippet_words)
			if text is None:
				continue
			words = text.split()
			if len(words) < snippet_words:
				continue

			latest_timestamp = fetch_latest_timestamp(title)
			if latest_timestamp is None:
				continue
			latest_dt = datetime.fromisoformat(latest_timestamp.replace("Z", "+00:00"))
			if latest_dt.year < min_year:
				continue  # favor relatively recent material

			creation_timestamp = fetch_creation_timestamp(title)
			if creation_timestamp is None:
				continue
			creation_dt = datetime.fromisoformat(creation_timestamp.replace("Z", "+00:00"))

			label = 1 if creation_dt.year < 2025 else 0
			if label == 0:
				print(f"Article '{title}' is labeled 0 (created in {creation_dt.year}).")
				remaining_quota = max(0, desired_label_zero - label_counts[0])
				if remaining_quota <= 0:
					continue
				available_chunks = len(words) // snippet_words
				chunks_to_use = min(CHUNKS_PER_LABEL_ZERO, available_chunks, remaining_quota)
				if chunks_to_use <= 0:
					continue
				for chunk_idx in range(chunks_to_use):
					start = chunk_idx * snippet_words
					end = start + snippet_words
					snippet = " ".join(words[start:end])
					collected.append(
						ArticleSample(
							title=title,
							input_text=snippet,
							label=0,
							created_at=creation_timestamp,
							last_modified=latest_timestamp,
						)
					)
				label_counts[0] += chunks_to_use
				print(f"Added {chunks_to_use} chunks for label 0 (total now {label_counts[0]}).")
				break
			else:
				if label_counts[1] >= desired_label_one:
					continue
				snippet = " ".join(words[:snippet_words])
				collected.append(
					ArticleSample(
						title=title,
						input_text=snippet,
						label=1,
						created_at=creation_timestamp,
						last_modified=latest_timestamp,
					)
				)
				label_counts[1] += 1
				print(f"Added 1 chunk for label 1 (total now {label_counts[1]}).")
			if goals_met():
				break

		if goals_met():
			break

	if len(collected) < min_total:
		raise RuntimeError(
			"Collected {total} samples (label0={label0}, label1={label1}) after {attempts} attempts; "
			"try adjusting parameters or increasing max attempts."
			.format(total=len(collected), label0=label_counts[0], label1=label_counts[1], attempts=attempts)
		)

	if abs(label_counts[0] - label_counts[1]) > BALANCE_TOLERANCE:
		print(
			f"Warning: label imbalance remains (label0={label_counts[0]}, label1={label_counts[1]}). "
			"Consider increasing --count or retries for better balance."
		)

	random.shuffle(collected)
	return collected


def save_outputs(samples: Iterable[ArticleSample], output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	jsonl_path = output_dir / "recent_wikimia.jsonl"
	records = [sample.to_dict() for sample in samples]

	with jsonl_path.open("w", encoding="utf-8") as f:
		for record in records:
			f.write(json.dumps(record, ensure_ascii=False) + "\n")

	dataset = Dataset.from_list(records)
	dataset.save_to_disk(str(output_dir / "hf_dataset"))
	print(f"Saved JSONL to {jsonl_path}")
	print(f"Saved Hugging Face dataset to {output_dir / 'hf_dataset'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compile a recent WikiMIA-style dataset from Wikipedia.")
	parser.add_argument("--count", type=int, default=500, help="Number of distinct articles/snippets to collect.")
	parser.add_argument("--snippet_words", type=int, default=32, help="Number of words per snippet.")
	parser.add_argument(
		"--min_year",
		type=int,
		default=2015,
		help="Ignore articles whose latest revision predates this year to favor recent coverage.",
	)
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=Path("data/recent_wikimia"),
		help="Directory where JSONL and HF dataset artifacts will be stored.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	samples = build_samples(
		target_count=args.count,
		snippet_words=args.snippet_words,
		min_year=args.min_year,
	)
	save_outputs(samples, args.output_dir)
	print(f"Collected {len(samples)} samples with {args.snippet_words}-word snippets.")


if __name__ == "__main__":
	main()

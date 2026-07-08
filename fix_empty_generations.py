from dotenv import load_dotenv

load_dotenv()

import argparse
import os
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

from utils.prompt_utils import get_wp_prompts, get_reuter_prompts, get_essay_prompts

# uv run python fix_empty_generations.py --reuter_gpt_plain
# uv run python fix_empty_generations.py --wp_gpt_plain 
# uv run python fix_empty_generations.py --essay_gpt_plain
# uv run python fix_empty_generations.py --reuter_gpt_plain --debug

# --- Copied straight from generate.py so the regenerated file uses the exact
# same prompt index as the original run. The prompt wording itself lives in
# utils/prompt_utils.py, shared with generate.py, so both stay in sync. ---

PROMPT_TYPE_INDICES = {
    "gpt": 0,
    "gpt_prompt1": 1,
    "gpt_prompt2": 2,
    "gpt_writing": 3,
    "gpt_semantic": 4,
    "gpt_plain": 6,
}
ALL_TYPES = list(PROMPT_TYPE_INDICES.keys()) + ["claude"]


def prompt_index_for_type(type_):
    return 0 if type_ == "claude" else PROMPT_TYPE_INDICES[type_]


def round_to_100(n):
    return int(round(n / 100.0)) * 100

# --- LLM calling, mirroring generate.py's call_llm/openai_backoff/claude_backoff,
# but with lazily-created clients so a --debug or check-only run never needs
# API keys. ---

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai

        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return _get_openai_client().chat.completions.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def claude_backoff(**kwargs):
    if _anthropic is None:
        raise ImportError("anthropic package not installed; run: pip install anthropic")
    client = _anthropic.Anthropic()
    return client.messages.create(**kwargs)


def call_llm(messages, mode, model, debug=False):
    if debug:
        return "[DEBUG]"
    if mode == "gpt":
        response = openai_backoff(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    elif mode == "claude":
        response = claude_backoff(model=model, max_tokens=2048, messages=messages)
        return response.content[0].text.strip()
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'gpt' or 'claude'")


def strip_reuter_essay_boilerplate(reply):
    """Same post-processing generate.py applies to reuter/essay replies."""
    reply = reply.replace("\n\n", "\n")

    lines = reply.split("\n")
    if any(i in lines[0].lower() for i in ["sure", "certainly"]):
        reply = "\n".join(lines[1:])

    lines = reply.split("\n")
    if any(i in lines[0].lower() for i in ["title"]):
        reply = "\n".join(lines[1:])

    return reply


# --- Emptiness check, mirroring analyse_txt_folder.py's _emptiness ---

EXCLUDE_DIRS = {"logprobs", "headlines"}


def is_empty(path: Path) -> bool:
    if path.stat().st_size == 0:
        return True
    return path.read_text(encoding="utf-8", errors="replace").strip() == ""


def find_empty_files(root: Path):
    if not root.is_dir():
        raise SystemExit(f"No such directory: {root}")

    empty = []
    for f in sorted(root.rglob("*.txt")):
        if not EXCLUDE_DIRS.isdisjoint(f.relative_to(root).parts):
            continue
        if is_empty(f):
            empty.append(f)
    return empty


# --- Per-dataset regeneration: figure out the right prompt + words for a
# given empty file, faithfully reproducing the corresponding block in
# generate.py. ---


def regenerate_wp(path: Path, dataset, type_, mode, model, debug):
    idx = path.stem  # e.g. "15"
    prompt = Path(f"data/{dataset}/prompts/{idx}.txt").read_text().strip()
    words = round_to_100(len(Path(f"data/{dataset}/human/{idx}.txt").read_text().split(" ")))

    prompts = get_wp_prompts(words, prompt)
    variant_prompt = prompts[prompt_index_for_type(type_)]

    reply = call_llm(
        messages=[{"role": "user", "content": variant_prompt}],
        mode=mode,
        model=model,
        debug=debug,
    )
    return reply.replace("\n\n", "\n")


def regenerate_essay(path: Path, type_, mode, model, debug):
    idx = path.stem
    prompt = Path(f"data/essay/prompts/{idx}.txt").read_text().strip()
    words = round_to_100(len(Path(f"data/essay/human/{idx}.txt").read_text().split(" ")))

    prompts = get_essay_prompts(words, prompt)
    variant_prompt = prompts[prompt_index_for_type(type_)]

    reply = call_llm(
        messages=[{"role": "user", "content": variant_prompt}],
        mode=mode,
        model=model,
        debug=debug,
    )
    if debug:
        return reply
    return strip_reuter_essay_boilerplate(reply)


def regenerate_reuter(path: Path, type_, mode, model, debug):
    author, idx = path.parts[-2], path.stem
    words = round_to_100(len(Path(f"data/reuter/human/{author}/{idx}.txt").read_text().split(" ")))
    # Headlines are always written under the `gpt` folder regardless of variant
    # (see generate.py's --reuter_prompts block), not under `type_`.
    headline = Path(f"data/reuter/gpt/{author}/headlines/{idx}.txt").read_text().strip()

    prompts = get_reuter_prompts(words, headline)
    variant_prompt = prompts[prompt_index_for_type(type_)]

    reply = call_llm(
        messages=[{"role": "user", "content": variant_prompt}],
        mode=mode,
        model=model,
        debug=debug,
    )
    if debug:
        return reply
    return strip_reuter_essay_boilerplate(reply)


def regenerate_one(path: Path, dataset, type_, mode, model, debug):
    if dataset == "wp":
        return regenerate_wp(path, dataset, type_, mode, model, debug)
    elif dataset == "essay":
        return regenerate_essay(path, type_, mode, model, debug)
    elif dataset == "reuter":
        return regenerate_reuter(path, type_, mode, model, debug)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")


def selected_dataset_type(args):
    selected = []
    for dataset in ("wp", "reuter", "essay"):
        for type_ in ALL_TYPES:
            if getattr(args, f"{dataset}_{type_}"):
                selected.append((dataset, type_))
    return selected


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find empty (0-byte or whitespace-only) generated .txt files under "
            "data/<dataset>/<type> and, on confirmation, regenerate only those "
            "files using the same prompt the original generate.py run would have used."
        )
    )

    for dataset in ("wp", "reuter", "essay"):
        for type_ in ALL_TYPES:
            parser.add_argument(f"--{dataset}_{type_}", action="store_true")

    parser.add_argument("--gpt_model", type=str, default="gpt-5.4-mini",
                        help="OpenAI model to use when regenerating (must match the original run's model)")
    parser.add_argument("--claude_model", type=str, default="claude-sonnet-5",
                        help="Anthropic model to use when regenerating (must match the original run's model)")
    parser.add_argument("--debug", action="store_true",
                        help="Don't call any real API; write the literal string '[DEBUG]' instead")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    selected = selected_dataset_type(args)
    if len(selected) != 1:
        parser.error(
            "Pass exactly one flag identifying the folder to check, e.g. --reuter_gpt_plain "
            "(got {}).".format(len(selected))
        )
    dataset, type_ = selected[0]

    root = Path(f"data/{dataset}/{type_}")
    empty_files = find_empty_files(root)

    if not empty_files:
        print(f"No empty .txt files found under {root}. Nothing to do.")
        return

    print(f"Found {len(empty_files)} empty file(s) under {root}:")
    for i, f in enumerate(empty_files, 1):
        print(f"  [{i}] {f.relative_to(root)}")

    if args.debug:
        print("\n[DEBUG MODE] Regeneration would write the literal string '[DEBUG]' instead of calling a real API.")

    answer = input(f"\nRegenerate these {len(empty_files)} file(s)? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("No action taken.")
        return

    mode = "claude" if type_ == "claude" else "gpt"
    model = args.claude_model if mode == "claude" else args.gpt_model

    regenerated, skipped, failed = [], [], []

    for f in empty_files:
        # Double-check right before regenerating: don't clobber a file that
        # got filled in (by this script or another process) since the scan above.
        if not is_empty(f):
            print(f"  SKIP  {f.relative_to(root)}: no longer empty, leaving it alone.")
            skipped.append(f)
            continue

        print(f"  Confirmed empty: {f.relative_to(root)} -- regenerating...")
        try:
            reply = regenerate_one(f, dataset, type_, mode, model, args.debug)
            f.write_text(reply)
            print(f"  OK    wrote {len(reply.split())} word(s) to {f.relative_to(root)}")
            regenerated.append(f)
        except Exception as e:
            print(f"  FAIL  {f.relative_to(root)}: {e}")
            failed.append((f, e))

    print("\n=== Summary ===")
    print(f"Regenerated: {len(regenerated)}")
    print(f"Skipped (already filled in): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for f, e in failed:
            print(f"  {f.relative_to(root)}: {e}")


if __name__ == "__main__":
    main()

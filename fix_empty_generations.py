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
from utils.env_utils import resolve_endpoint

# uv run python fix_empty_generations.py --reuter_gpt_plain
# uv run python fix_empty_generations.py --wp_gpt_plain 
# uv run python fix_empty_generations.py --essay_gpt_plain
# uv run python fix_empty_generations.py --reuter_gpt_plain --debug
# uv run python fix_empty_generations.py --reuter_gpt_plain --out_name gpt56luna_0701A

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


# ANSI, so a file that came back empty AGAIN cannot be mistaken for a success in
# a wall of green OK lines. Not conditional on a tty: run_full_pipeline.py prints
# its own colours through the same pipe.
YELLOW = "\033[33m"
RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"


# Must match generate.py's WORD_BUDGET_STEP / MIN_WORD_BUDGET exactly, or a
# refilled file is written to a different length than its neighbours were.
WORD_BUDGET_STEP = 50
MIN_WORD_BUDGET = 50


def round_to_50(n):
    """The word budget generate.py asks the model for, from the human length.

    A multiple of 50, never below 50. The floor is the point: this was
    round_to_100, and round() is HALF TO EVEN, so a 50-word human document gave
    a budget of 0 -- a prompt reading "write a news article in 0 words", which
    the model answered correctly by returning nothing. That is the failure this
    script was written to clean up after; with the floor it cannot happen.
    """
    return max(int(round(n / float(WORD_BUDGET_STEP))) * WORD_BUDGET_STEP, MIN_WORD_BUDGET)


def word_budget(path: Path, dataset):
    """The budget this file's regeneration would ask for, without calling anything.

    Same expression as the regenerate_* functions below, pulled out so main() can
    report the budget per file before spending anything, and can tell a file with
    no human original at all (None) from one that is merely short.
    """
    if dataset == "reuter":
        author, idx = path.parts[-2], path.stem
        human = Path(f"data/reuter/human/{author}/{idx}.txt")
    else:
        human = Path(f"data/{dataset}/human/{path.stem}.txt")
    if not human.is_file():
        return None
    return round_to_50(len(human.read_text().split(" ")))

# --- LLM calling, mirroring generate.py's call_llm/openai_backoff/claude_backoff,
# but with lazily-created clients so a --debug or check-only run never needs
# API keys. ---

# Same hard ceiling on one reply as generate.py's MAX_OUTPUT_TOKENS, and for the
# same reason: this script re-calls the model for every blank file, so a model
# that will not stop itself (an open-weight model on a llama.cpp / Ollama
# backend falling into a paragraph-level loop) would run to the context window
# or a server timeout here just as it would in generate.py. A refill is capped
# to the same length its neighbours were.
MAX_OUTPUT_TOKENS = 4096

_openai_client = None
_openai_client_endpoint = None


def _get_openai_client(base_url=None, api_key_env="OPENAI_API_KEY"):
    """Lazily build (or rebuild, if the endpoint changed) the OpenAI client.

    Lazy so a --debug or check-only run never needs an API key. Keyed on the
    (base_url, api_key_env) pair rather than built once, because this script
    checks one (dataset, type) per invocation but nothing stops --api_key_env
    or --base_url differing between calls in a future caller.
    """
    global _openai_client, _openai_client_endpoint
    endpoint = (base_url, api_key_env)
    if _openai_client is None or _openai_client_endpoint != endpoint:
        import openai

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"{api_key_env} is not set")
        _openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        _openai_client_endpoint = endpoint
    return _openai_client


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(base_url=None, api_key_env="OPENAI_API_KEY", **kwargs):
    return _get_openai_client(base_url, api_key_env).chat.completions.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def claude_backoff(**kwargs):
    if _anthropic is None:
        raise ImportError("anthropic package not installed; run: pip install anthropic")
    client = _anthropic.Anthropic()
    return client.messages.create(**kwargs)


def call_llm(messages, mode, model, debug=False, base_url=None, api_key_env="OPENAI_API_KEY"):
    if debug:
        return "[DEBUG]"
    # `or ""`: a model that answers with no content at all returns None here, and
    # a None that reaches .strip() crashes the file with an AttributeError
    # instead of being reported as the empty reply it is.
    if mode == "gpt":
        response = openai_backoff(base_url=base_url, api_key_env=api_key_env,
                                  model=model, messages=messages,
                                  max_tokens=MAX_OUTPUT_TOKENS)
        return (response.choices[0].message.content or "").strip()
    elif mode == "claude":
        response = claude_backoff(model=model, max_tokens=2048, messages=messages)
        return (response.content[0].text if response.content else "").strip()
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'gpt' or 'claude'")


def strip_reuter_essay_boilerplate(reply, drop_preamble=False):
    """Same post-processing generate.py applies to reuter/essay replies.

    Must stay in step with ``generate.strip_boilerplate``, including the
    default: a refill post-processed differently from its neighbours is a
    document that does not belong in the corpus it sits in.
    """
    reply = reply.replace("\n\n", "\n")

    if not drop_preamble:
        return reply

    original = reply

    lines = reply.split("\n")
    if any(i in lines[0].lower() for i in ["sure", "certainly"]):
        reply = "\n".join(lines[1:])

    lines = reply.split("\n")
    if any(i in lines[0].lower() for i in ["title"]):
        reply = "\n".join(lines[1:])

    # See generate.strip_boilerplate: dropping line 0 of a single-paragraph
    # reply drops the document. Returning the unstripped reply is what stops
    # this script looping forever on a file it can never fill.
    if not reply.strip():
        return original

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


def regenerate_wp(path: Path, dataset, type_, mode, model, debug, words,
                  base_url=None, api_key_env="OPENAI_API_KEY"):
    idx = path.stem  # e.g. "15"
    prompt = Path(f"data/{dataset}/prompts/{idx}.txt").read_text().strip()

    prompts = get_wp_prompts(words, prompt)
    variant_prompt = prompts[prompt_index_for_type(type_)]

    reply = call_llm(
        messages=[{"role": "user", "content": variant_prompt}],
        mode=mode,
        model=model,
        debug=debug,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    # (raw, cleaned): the raw reply is kept so a reply that survives the API and
    # is then deleted by post-processing can be told apart from one the model
    # never sent. Both end as an empty file, and they need different fixes.
    return reply, reply.replace("\n\n", "\n")


def regenerate_essay(path: Path, type_, mode, model, debug, words, drop_preamble=False,
                     base_url=None, api_key_env="OPENAI_API_KEY"):
    idx = path.stem
    prompt = Path(f"data/essay/prompts/{idx}.txt").read_text().strip()

    prompts = get_essay_prompts(words, prompt)
    variant_prompt = prompts[prompt_index_for_type(type_)]

    reply = call_llm(
        messages=[{"role": "user", "content": variant_prompt}],
        mode=mode,
        model=model,
        debug=debug,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    if debug:
        return reply, reply
    return reply, strip_reuter_essay_boilerplate(reply, drop_preamble)


def regenerate_reuter(path: Path, type_, mode, model, debug, words, drop_preamble=False,
                      base_url=None, api_key_env="OPENAI_API_KEY"):
    author, idx = path.parts[-2], path.stem
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
        base_url=base_url,
        api_key_env=api_key_env,
    )
    if debug:
        return reply, reply
    return reply, strip_reuter_essay_boilerplate(reply, drop_preamble)


def regenerate_one(path: Path, dataset, type_, mode, model, debug, words,
                   drop_preamble=False, base_url=None, api_key_env="OPENAI_API_KEY"):
    """``(raw_reply, text_to_write)`` for one file."""
    if dataset == "wp":
        # wp never had the line-dropping, so drop_preamble does not reach it.
        return regenerate_wp(path, dataset, type_, mode, model, debug, words,
                             base_url, api_key_env)
    elif dataset == "essay":
        return regenerate_essay(path, type_, mode, model, debug, words, drop_preamble,
                                base_url, api_key_env)
    elif dataset == "reuter":
        return regenerate_reuter(path, type_, mode, model, debug, words, drop_preamble,
                                 base_url, api_key_env)
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
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "genai4science"],
                        help="Which OpenAI-compatible host serves --gpt_model. "
                             "Must match the original generate.py run's "
                             "--provider, or the refill comes from a different "
                             "model. See generate.py --provider.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Same meaning as generate.py --base_url; must "
                             "match the original run. Paired with --api_key_env.")
    parser.add_argument("--api_key_env", type=str, default=None,
                        help="Env var holding the API key for --base_url.")
    parser.add_argument("--debug", action="store_true",
                        help="Don't call any real API; write the literal string '[DEBUG]' instead")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt. Use this for any "
                             "non-interactive run: the prompt is a bare input(), "
                             "so it reads whatever is queued on stdin -- the next "
                             "line of a pasted command block included.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any file is still empty afterwards. Off by "
                             "default: run_full_pipeline.py aborts on a non-zero "
                             "exit, and the unfixable zero-budget files are present "
                             "in every run.")
    parser.add_argument("--strip_boilerplate", action="store_true",
                        help="Drop the first line of a reuter/essay reply when it "
                             "contains 'sure', 'certainly' or 'title'. OFF BY "
                             "DEFAULT, and must match the flag the original "
                             "generate.py run used, or the refill is "
                             "post-processed differently from its neighbours. "
                             "See generate.py --strip_boilerplate for why it is "
                             "off.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Look under data/<dataset>/<OUT_NAME>/ instead of data/<dataset>/<type>/, "
                             "matching generate.py --out_name. The prompt is still chosen by the "
                             "--<dataset>_<type> flag, so the refill uses the same prompt as the "
                             "original run.")

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

    root = Path(f"data/{dataset}/{args.out_name or type_}")
    empty_files = find_empty_files(root)

    if not empty_files:
        print(f"{GREEN}No empty .txt files found under {root}. Nothing to do.{RESET}")
        return 0

    # The budget first, for every file: a zero one is decided by the human
    # original and no number of retries changes it, so those files are reported
    # and taken out rather than paid for.
    budgets = {f: word_budget(f, dataset) for f in empty_files}
    # Kept as a guard, not as an expected case: round_to_50 floors at
    # MIN_WORD_BUDGET, so nothing can ask for zero words any more. If this bucket
    # is ever non-empty, the floor has been removed or bypassed -- which is
    # exactly the regression worth being told about before paying for a run.
    hopeless = [f for f, w in budgets.items() if w is not None and w < MIN_WORD_BUDGET]
    unknown = [f for f, w in budgets.items() if w is None]
    retryable = [f for f in empty_files if f not in hopeless and f not in unknown]

    print(f"Found {len(empty_files)} empty file(s) under {root}:")
    for i, f in enumerate(empty_files, 1):
        words = budgets[f]
        if words is None:
            note = f"  {RED}no human original to size it against{RESET}"
        elif words < MIN_WORD_BUDGET:
            note = f"  {YELLOW}<- asks the model for {words} words{RESET}"
        else:
            note = f"  {DIM}(asks for {words} words){RESET}"
        print(f"  [{i}] {str(f.relative_to(root)):<28}{note}")

    if hopeless:
        print(
            f"\n{RED}{len(hopeless)} file(s) would ask for fewer than "
            f"{MIN_WORD_BUDGET} words.{RESET}\n"
            f"  That should be impossible: round_to_50() floors the budget at\n"
            f"  MIN_WORD_BUDGET, precisely so a short human original cannot produce a\n"
            f"  \"write ... in 0 words\" prompt. Check that generate.py and this script\n"
            f"  still agree on WORD_BUDGET_STEP and MIN_WORD_BUDGET before re-running."
        )
    if unknown:
        print(f"\n{RED}{len(unknown)} file(s) have no human original at all:{RESET}")
        for f in unknown:
            print(f"  {f.relative_to(root)}")

    if not retryable:
        print(f"\n{YELLOW}Nothing left to try.{RESET} No API call was made.")
        return 1 if args.strict else 0

    if args.debug:
        print("\n[DEBUG MODE] Regeneration would write the literal string '[DEBUG]' instead of calling a real API.")

    if args.yes:
        print(f"\nRegenerating {len(retryable)} file(s) (--yes).")
    else:
        try:
            answer = input(
                f"\nRegenerate these {len(retryable)} file(s)? [y/N]: "
            ).strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            # Say what was read. A bare input() takes whatever is queued on
            # stdin, and the usual source of a surprise "no" is the next line of
            # a pasted command block arriving while this script was still
            # running -- which looks identical to a deliberate refusal.
            print(f"{YELLOW}No action taken.{RESET} (read {answer!r} — if you did "
                  f"not type that, stdin had a queued line: paste one command at "
                  f"a time, or pass --yes.)")
            return 0

    mode = "claude" if type_ == "claude" else "gpt"
    model = args.claude_model if mode == "claude" else args.gpt_model

    base_url = api_key_env = None
    if mode == "gpt" and not args.debug:
        base_url, api_key_env = resolve_endpoint(args.provider, args.base_url, args.api_key_env)
        if not os.environ.get(api_key_env):
            parser.error(
                f"{api_key_env} is not set -- required by --gpt_model {model!r} "
                f"(--provider {args.provider!r}). Must match the original run."
            )

    regenerated, skipped, failed, still_empty = [], [], [], []

    for f in retryable:
        # Double-check right before regenerating: don't clobber a file that
        # got filled in (by this script or another process) since the scan above.
        if not is_empty(f):
            print(f"  SKIP  {f.relative_to(root)}: no longer empty, leaving it alone.")
            skipped.append(f)
            continue

        print(f"  Confirmed empty: {f.relative_to(root)} -- regenerating...")
        try:
            raw, text = regenerate_one(
                f, dataset, type_, mode, model, args.debug, budgets[f],
                args.strip_boilerplate, base_url, api_key_env,
            )
        except Exception as e:
            print(f"  {RED}FAIL{RESET}  {f.relative_to(root)}: {e}")
            failed.append((f, e))
            continue

        # An empty result is the failure this script exists to find, so it is
        # never written and never counted as a success. Writing it would leave
        # the file exactly as it was while the summary claimed otherwise.
        if not text.strip():
            if raw.strip():
                # The API answered; post-processing removed all of it. Kept as a
                # guard, not as an expected case: the only step that could do
                # this was the preamble drop, which is now off by default and
                # returns the unstripped reply rather than nothing when it would
                # empty a document. Reaching this means that guard is gone.
                print(f"  {YELLOW}EMPTY{RESET} {f.relative_to(root)}: the model "
                      f"replied {len(raw.split())} word(s), but post-processing "
                      f"removed all of it.\n"
                      f"        raw reply: {raw.strip()[:200]!r}\n"
                      f"        That should no longer be possible: "
                      f"strip_reuter_essay_boilerplate() only drops the first "
                      f"line under --strip_boilerplate, and never returns an "
                      f"empty document. Check that guard before re-running.")
            else:
                print(f"  {YELLOW}EMPTY{RESET} {f.relative_to(root)}: the model "
                      f"returned nothing for a {budgets[f]}-word request. "
                      f"File left as it was.")
            still_empty.append(f)
            continue

        f.write_text(text)
        print(f"  {GREEN}OK{RESET}    wrote {len(text.split())} word(s) to "
              f"{f.relative_to(root)}")
        regenerated.append(f)

    print("\n=== Summary ===")
    print(f"{GREEN}Regenerated: {len(regenerated)}{RESET}")
    print(f"Skipped (already filled in): {len(skipped)}")
    if still_empty:
        print(f"{YELLOW}Still empty (model returned nothing): {len(still_empty)}{RESET}")
        for f in still_empty:
            print(f"  {f.relative_to(root)}")
    else:
        print("Still empty (model returned nothing): 0")
    if hopeless:
        print(f"{RED}Not attempted (budget below {MIN_WORD_BUDGET}): "
              f"{len(hopeless)}{RESET}")
        for f in hopeless:
            print(f"  {f.relative_to(root)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for f, e in failed:
            print(f"  {f.relative_to(root)}: {e}")

    # 0 unless asked to be strict: run_full_pipeline.py aborts the whole run on a
    # non-zero exit here, and a file the model simply declined to fill must not
    # stop a pipeline that has nothing else wrong.
    if args.strict and (still_empty or hopeless or failed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

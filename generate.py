from dotenv import load_dotenv

# Load .env before building the OpenAI/Anthropic clients below so OPENAI_API_KEY /
# ANTHROPIC_API_KEY are already in the environment.
load_dotenv()

import argparse
import openai
import re
import tqdm
import os
import math
import nltk
import numpy as np
import string
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

# python generate.py --debug --limit 3           # mock LLM calls, first 3 items
# uv run python generate.py --reuter_prompts --limit 10 
# uv run python generate.py --reuter_gpt_plain --limit 10 
# uv run python generate.py --wp_gpt_plain --limit 10  
# uv run python generate.py --essay_gpt_plain --limit 10
# uv run python generate.py --essay_gpt_plain --out_name gpt56luna_0701A --limit 10
#   ^ same prompt, but written to data/essay/gpt56luna_0701A/ so it cannot be
#     confused with (or skipped because of) another model's gpt_plain run.


try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

from nltk.corpus import wordnet
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForCausalLM

from utils.generate import generate_documents
from utils.write_logprobs import write_logprobs, write_llama_logprobs
from utils.symbolic import convert_file_to_logprob_file
from utils.load import Dataset, get_generate_dataset
from utils.prompt_utils import get_wp_prompts, get_reuter_prompts, get_essay_prompts
from utils.env_utils import resolve_endpoint


nltk.download("wordnet")
nltk.download("omw-1.4")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

datasets = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]
generate_dataset_fn = get_generate_dataset(*datasets)

# Maps each gpt_* prompt type to its index in the list returned by get_*_prompts().
PROMPT_TYPE_INDICES = {
    "gpt": 0,
    "gpt_prompt1": 1,
    "gpt_prompt2": 2,
    "gpt_writing": 3,
    "gpt_semantic": 4,
    "gpt_plain": 6,
}
# Claude only generates the default (unstyled) prompt, not the gpt_prompt1/gpt_prompt2/
# gpt_writing/gpt_semantic variants, so it always uses prompts[0], the plain prompt each
# get_*_prompts() returns first.
prompt_types_claude = ["claude"]


def prompt_index_for_type(type):
    return 0 if type == "claude" else PROMPT_TYPE_INDICES[type]


def selected_gpt_types(args, prefix):
    """Return the prompt-type names whose --{prefix}_{type} flag was passed."""
    return [type for type in PROMPT_TYPE_INDICES if getattr(args, f"{prefix}_{type}")]


def out_dir_for_type(type, args):
    """The directory under data/<dataset>/ that this type's documents go in.

    Defaults to the type name, which is what every original invocation expects.
    --out_name overrides it so a run can say which MODEL wrote the documents as
    well as which prompt: the type name alone is the same for gpt-3.5 and
    gpt-5.6, and since every loop below skips an output file that already
    exists, a second model generated into the first one's directory writes
    nothing at all and looks like a complete run.

    Only the output path changes. The prompt is still chosen by `type`, so
    --out_name cannot alter what is sent to the model.
    """
    return args.out_name or type


html_replacements = [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
]

perturb_char_names = [
    "char_basic",
    "char_space",
    "char_cap",
    "word_adj",
    "word_syn",
]
perturb_char_sizes = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

perturb_sent_names = ["sent_adj", "sent_paraph", "para_adj", "para_paraph"]
perturb_sent_sizes = list(range(11))


def closest_synonym(word):
    synonyms = wordnet.synsets(word)
    if not synonyms:
        return None  # Return None if there are no synonyms
    closest_synset = synonyms[0]  # Assume the first synset is the closest
    for synset in synonyms[1:]:
        # Update closest_synset if we find a synset with more lemmas (synonyms)
        if len(synset.lemmas()) > len(closest_synset.lemmas()):
            closest_synset = synset
    # Return the name of the lemma from the closest synset
    # that is not the same as the input word
    for lemma in closest_synset.lemmas():
        if lemma.name() != word:
            return lemma.name()
    return None


def html_replace(text):
    for replacement in html_replacements:
        text = text.replace(replacement[0], replacement[1])
    return text


# Built in __main__, once the endpoint is resolved from --provider /
# --base_url / --api_key_env -- so a single invocation (one model, one
# endpoint, per the --out_name contract already documented above) needs one
# client, built once, before any worker thread can read it.
_openai_client = None

# Hard ceiling on one reply. Every prompt asks for an essay / story / article of
# a stated word count (<= ~1000 words, so <= ~1500 tokens); 4096 never clips a
# real one. It is a stop for a model that will not stop itself: an open-weight
# model on a llama.cpp / Ollama backend with no repetition penalty can fall into
# a paragraph-level loop -- re-writing its own conclusion dozens of times -- and
# run to the context window or a server timeout. Same reason the claude path
# below already caps at 2048.
MAX_OUTPUT_TOKENS = 4096


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return _openai_client.chat.completions.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def claude_backoff(**kwargs):
    if _anthropic is None:
        raise ImportError("anthropic package not installed; run: pip install anthropic")
    client = _anthropic.Anthropic()
    return client.messages.create(**kwargs)


def call_llm(messages, mode, model, debug=False):
    """Call OpenAI or Anthropic depending on `mode` ("gpt" or "claude")."""
    if debug:
        return "[DEBUG] This is a mock LLM response used for testing."
    # `or ""` on both branches: a model that answers with no content at all
    # returns None (OpenAI) or an empty content list (Anthropic), and a None
    # reaching .strip() raises AttributeError -- which the caller records as a
    # failed call with no file written, hiding an empty REPLY behind what looks
    # like a transport error. An empty string is what actually happened.
    if mode == "gpt":
        response = openai_backoff(
            model=model, messages=messages, max_tokens=MAX_OUTPUT_TOKENS
        )
        return (response.choices[0].message.content or "").strip()
    elif mode == "claude":
        response = claude_backoff(
            model=model,
            max_tokens=2048,
            messages=messages,
        )
        return (response.content[0].text if response.content else "").strip()
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'gpt' or 'claude'")


def strip_boilerplate(reply, drop_preamble=False):
    """Collapse blank lines, and optionally drop a preamble or title line.

    Shared by the reuter and essay loops, which applied identical copies of it
    inline. ``fix_empty_generations.py`` carries the same function under the
    name ``strip_reuter_essay_boilerplate``; both must stay in step, or a
    refilled file is post-processed differently from its neighbours.

    THE BLANK-LINE COLLAPSE IS UNCONDITIONAL. Only the line-dropping below is
    behind ``drop_preamble``, and it defaults to OFF because it does far more
    harm than good on this corpus -- see the flag's help in the parser.
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

    # Never hand back nothing. A single-paragraph reply is ONE line here (the
    # collapse above joined its paragraphs), so dropping line 0 drops the whole
    # document -- which is how a 152-word essay became a 0-byte file that
    # fix_empty_generations.py could then never refill, because it re-ran the
    # same stripper and got the same empty string. This is a deliberate
    # deviation from upstream, which wrote the empty string out.
    if not reply.strip():
        return original

    return reply


def make_call(content, mode, model, debug, post=None):
    """Build a zero-argument thunk returning the text to write for one document.

    Everything the call depends on -- the finished prompt string, the model, the
    post-processing -- is captured here, on the main thread, so the thunk itself
    reads no shared state. That is what makes it safe to hand to a worker: two
    calls in flight share nothing but the HTTP client, which is thread-safe, and
    the API itself is stateless, so neither can see the other's content.
    """

    def _call():
        reply = call_llm(
            messages=[{"role": "user", "content": content}],
            mode=mode,
            model=model,
            debug=debug,
        )
        return post(reply) if post else reply

    return _call


def run_parallel(tasks, workers, desc, stats):
    """Run ``(out_path, thunk)`` tasks concurrently, writing each result as it lands.

    **Only the network call is parallel.** The worker returns text and does
    nothing else; the file write and the ``stats`` bookkeeping happen here, on
    the thread consuming ``as_completed``, so there is no shared mutable state
    to guard and no lock to get wrong. Results are matched by future identity
    (``futures[fut]``), never by completion order, so a reply cannot be filed
    under another document's path.

    The task list is built by the caller *before* any submission, which is where
    the skip-if-exists check and every ``os.makedirs`` belong: both are
    read-then-act sequences that race if two threads run them at once.

    Ctrl-C cancels what has not started instead of waiting for it. A bare
    ``with ThreadPoolExecutor(...)`` calls shutdown(wait=True) on the way out,
    so an interrupt with a thousand tasks queued appears to hang.
    """
    tasks = list(tasks)
    if not tasks:
        return

    # Two tasks writing one path would both run and both write, and the file
    # would hold whichever reply finished last -- a paid-for document silently
    # discarded. The loops below cannot produce one (a path is (type, index),
    # and --out_name is restricted to a single type), so this is a guard against
    # a future loop, not a known case.
    seen, repeated = set(), set()
    for out_path, _ in tasks:
        if out_path in seen:
            repeated.add(out_path)
        seen.add(out_path)
    if repeated:
        duplicates = sorted(repeated)
        raise ValueError(
            f"{len(duplicates)} output path(s) claimed by more than one task, "
            f"e.g. {duplicates[:3]}. Each document must have its own path."
        )

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = {pool.submit(thunk): out_path for out_path, thunk in tasks}
    try:
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc=desc):
            out_path = futures[fut]
            try:
                text = fut.result()
                with open(out_path, "w") as f:
                    f.write(text)
            except Exception as e:
                record_result(stats, out_path, error=e)
                continue
            if not text.strip():
                # The file is written either way, so fix_empty_generations.py can
                # find and refill it -- but it is not a document, and counting it
                # as one is how a run reports 1000 created and yields 999.
                record_result(stats, out_path, error="empty reply")
            else:
                record_result(stats, out_path)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def record_result(stats, path, error=None):
    """Track a generated file's outcome for the end-of-run summary."""
    if error is None:
        stats["created"].append(path)
    else:
        stats["missed"].append((path, str(error)))


def group_counts_by_dir(paths):
    counts = {}
    for path in paths:
        folder = os.path.dirname(path)
        counts[folder] = counts.get(folder, 0) + 1
    return counts


def print_and_log_summary(stats, log_path="generate.log"):
    if not stats["created"] and not stats["missed"]:
        return

    lines = [
        "",
        "=== Generation Summary ===",
        f"Created: {len(stats['created'])}",
        f"Missed:  {len(stats['missed'])}",
    ]

    if stats["created"]:
        lines.append("")
        lines.append("Created by folder:")
        for folder, count in sorted(group_counts_by_dir(stats["created"]).items()):
            lines.append(f"  {folder}: {count}")

    if stats["missed"]:
        lines.append("")
        lines.append("Missed by folder:")
        missed_paths = [path for path, _ in stats["missed"]]
        for folder, count in sorted(group_counts_by_dir(missed_paths).items()):
            lines.append(f"  {folder}: {count}")

        lines.append("")
        lines.append("Missed files:")
        # Sorted, not in the order they were recorded: with --workers > 1 that
        # order is whichever call finished first, so two runs over the same
        # failures produce logs that cannot be diffed.
        for path, error in sorted(stats["missed"]):
            lines.append(f"  {path}: {error}")

    summary = "\n".join(lines)
    print(summary)

    with open(log_path, "a") as f:
        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] generate.py run\n")
        f.write(summary)
        f.write("\n")


# Default number of articles to consider per Reuters author when --limit isn't given.
REUTER_ARTICLES_PER_AUTHOR = 20


# The word budget asked of the model is the human partner's length, rounded to
# this step and never allowed below it. Both numbers are 50 deliberately:
#
#   THE STEP was 100, which is coarse enough to matter at the short end -- a
#   450-word article and a 549-word one were both asked for 500. 50 tracks the
#   human length about twice as closely, at no cost.
#
#   THE FLOOR is what stops the bug this replaces. round() is HALF TO EVEN, so
#   round_to_100(50) was 0, not 100, and so was anything shorter: the prompt
#   read "write a news article in 0 words", which the model answered correctly
#   by returning nothing. One reuter article (AaronPressman/16, 50 tokens by
#   split(" ")) and the six blank essay seeds landed on exactly that. Half-to-
#   even still applies here -- round_to_50(75) is 100 -- but with a floor no
#   input can reach zero, so the failure cannot recur.
WORD_BUDGET_STEP = 50
MIN_WORD_BUDGET = 50


def round_to_50(n):
    """The word budget for a human document of `n` words: a multiple of 50, >= 50."""
    return max(int(round(n / float(WORD_BUDGET_STEP))) * WORD_BUDGET_STEP, MIN_WORD_BUDGET)


def generate_logprobs(generate_dataset_fn, llama_7b_model=None, llama_13b_model=None):
    files = generate_dataset_fn(lambda f: f)

    for file in tqdm.tqdm(files):
        if "logprobs" in file:
            continue

        base_path = os.path.dirname(file) + "/logprobs"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        with open(file, "r") as f:
            doc = f.read().strip()

        davinci_file = convert_file_to_logprob_file(file, "davinci")
        if not os.path.exists(davinci_file):
            write_logprobs(doc, davinci_file, "davinci")

        ada_file = convert_file_to_logprob_file(file, "ada")
        if not os.path.exists(ada_file):
            write_logprobs(doc, ada_file, "ada")

        llama_7b_file = convert_file_to_logprob_file(file, "llama-7b")
        if llama_7b_model and not os.path.exists(llama_7b_file):
            write_llama_logprobs(doc, llama_7b_file, llama_7b_model)

        llama_13b_file = convert_file_to_logprob_file(file, "llama-13b")
        if llama_13b_model and not os.path.exists(llama_13b_file):
            write_llama_logprobs(doc, llama_13b_file, llama_13b_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpt_model", type=str, default="gpt-5.4-mini",
                        help="OpenAI model to use for generation")
    parser.add_argument("--claude_model", type=str, default="claude-sonnet-5",
                        help="Anthropic Claude model to use instead of OpenAI")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "genai4science"],
                        help="Which OpenAI-compatible host serves --gpt_model. "
                             "Omitted or 'openai' calls OpenAI itself with "
                             "OPENAI_API_KEY, unchanged from before this flag "
                             "existed. 'genai4science' calls HUN-REN SZTAKI's "
                             "endpoint (GENAI4SCIENCE_API_KEY, or "
                             "GENAI4SCIENCE_PERFORMANCE_API_KEY when set, which "
                             "wins). Ignored for --*_claude.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Call this OpenAI-compatible base URL instead of "
                             "OpenAI's own, for a host --provider has no "
                             "shorthand for. Must be given together with "
                             "--api_key_env.")
    parser.add_argument("--api_key_env", type=str, default=None,
                        help="Env var holding the API key for --base_url.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: mock all LLM calls instead of making real API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap every loop to the first N items. Works with or without --debug, "
                             "so it can be used to test real API calls on a small sample.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API calls (default: 8). Each call is an independent, "
                             "stateless request built before submission, so concurrency cannot "
                             "mix one document's inputs into another's. Use 1 to make the run "
                             "strictly sequential; raise it only as far as the account's rate "
                             "limit allows, since a 429 costs a retry with backoff.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Write the generated documents to data/<dataset>/<OUT_NAME>/ instead of "
                             "data/<dataset>/<type>/. Use it to keep one model's output apart from "
                             "another's, e.g. --essay_gpt_plain --out_name gpt56luna_0701A. The prompt "
                             "is still selected by the --<dataset>_<type> flag, so this only renames the "
                             "output directory. Exactly one type may be selected when it is given.")

    parser.add_argument("--wp_prompts", action="store_true")
    parser.add_argument("--wp_human", action="store_true")
    parser.add_argument("--wp_gpt", action="store_true", help="Generate the main (unstyled) WP GPT prompt")
    parser.add_argument("--wp_gpt_prompt1", action="store_true", help="Generate the WP gpt_prompt1 variant")
    parser.add_argument("--wp_gpt_prompt2", action="store_true", help="Generate the WP gpt_prompt2 variant")
    parser.add_argument("--wp_gpt_writing", action="store_true", help="Generate the WP gpt_writing variant")
    parser.add_argument("--wp_gpt_semantic", action="store_true", help="Generate the WP gpt_semantic variant")
    parser.add_argument("--wp_gpt_plain", action="store_true", help="Generate the WP gpt_plain variant (no markdown formatting, for AI-detector datasets)")
    parser.add_argument("--wp_claude", action="store_true")

    parser.add_argument("--reuter_human", action="store_true")
    parser.add_argument("--reuter_prompts", action="store_true", help="Generate Reuters headlines from human articles, used as prompts")
    parser.add_argument("--reuter_gpt", action="store_true", help="Generate the main (unstyled) Reuters GPT prompt")
    parser.add_argument("--reuter_gpt_prompt1", action="store_true", help="Generate the Reuters gpt_prompt1 variant")
    parser.add_argument("--reuter_gpt_prompt2", action="store_true", help="Generate the Reuters gpt_prompt2 variant")
    parser.add_argument("--reuter_gpt_writing", action="store_true", help="Generate the Reuters gpt_writing variant")
    parser.add_argument("--reuter_gpt_semantic", action="store_true", help="Generate the Reuters gpt_semantic variant")
    parser.add_argument("--reuter_gpt_plain", action="store_true", help="Generate the Reuters gpt_plain variant (no markdown formatting, for AI-detector datasets)")
    parser.add_argument("--reuter_claude", action="store_true")

    parser.add_argument("--essay_prompts", action="store_true")
    parser.add_argument("--essay_human", action="store_true")
    parser.add_argument("--essay_gpt", action="store_true", help="Generate the main (unstyled) essay GPT prompt")
    parser.add_argument("--essay_gpt_prompt1", action="store_true", help="Generate the essay gpt_prompt1 variant")
    parser.add_argument("--essay_gpt_prompt2", action="store_true", help="Generate the essay gpt_prompt2 variant")
    parser.add_argument("--essay_gpt_writing", action="store_true", help="Generate the essay gpt_writing variant")
    parser.add_argument("--essay_gpt_semantic", action="store_true", help="Generate the essay gpt_semantic variant")
    parser.add_argument("--essay_gpt_plain", action="store_true", help="Generate the essay gpt_plain variant (no markdown formatting, for AI-detector datasets)")
    parser.add_argument("--essay_claude", action="store_true")

    parser.add_argument(
        "--strip_boilerplate", action="store_true",
        help="Drop the first line of a reuter/essay reply when it contains "
             "'sure', 'certainly' or 'title'. OFF BY DEFAULT, and best left "
             "off: the test is a SUBSTRING match on the whole first line, so "
             "it fires on 'ensure', 'measure', 'pressure' and 'entitled', and "
             "after blank-line collapsing the first line of a "
             "single-paragraph reply IS the whole document. Measured against "
             "the human corpus, roughly 9%% of essay and 3%% of reuter first "
             "paragraphs contain one of those substrings. This was upstream "
             "Ghostbuster's behaviour and is kept only to reproduce corpora "
             "built before it was made optional.")

    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--logprob_other", action="store_true")
    parser.add_argument("--logprob_llama", action="store_true")

    parser.add_argument("--gen_perturb_char", action="store_true")
    parser.add_argument("--logprob_perturb_char", action="store_true")

    parser.add_argument("--gen_perturb_sent", action="store_true")
    parser.add_argument("--logprob_perturb_sent", action="store_true")

    args = parser.parse_args()

    # Resolved once, here, and used for every --gpt_model call this invocation
    # makes -- consistent with --out_name already restricting one invocation to
    # one generator type. Skipped under --debug: no LLM call is ever made, so no
    # key is needed either (this is a relaxation of the previous behaviour,
    # which built an OpenAI client -- unused under --debug -- unconditionally).
    if not args.debug:
        base_url, api_key_env = resolve_endpoint(
            args.provider, args.base_url, args.api_key_env
        )
        api_key = os.environ.get(api_key_env)
        if not api_key:
            parser.error(
                f"{api_key_env} is not set -- required by --gpt_model "
                f"{args.gpt_model!r} (--provider {args.provider!r})"
            )
        _openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # --out_name names ONE directory, so it cannot describe two prompts. Two
    # types selected alongside it would either share a directory (and the
    # skip-if-exists logic would give all of it to whichever ran first) or need
    # a name each, which is what running the script twice is for.
    if args.out_name is not None:
        selected_types = set()
        for prefix in ("wp", "reuter", "essay"):
            selected_types.update(selected_gpt_types(args, prefix))
            if getattr(args, f"{prefix}_claude"):
                selected_types.add("claude")
        if len(selected_types) != 1:
            parser.error(
                "--out_name applies to a single generator type, but "
                f"{len(selected_types)} were selected ({sorted(selected_types) or 'none'}). "
                "Run once per type."
            )

    if args.debug:
        print("[DEBUG MODE] Mocking all LLM calls.")
    if args.limit is not None:
        print(f"Capping loops to the first {args.limit} items.")
    limit = args.limit  # None means use the original full size

    stats = {"created": [], "missed": []}

    if args.wp_prompts:

        def format_prompt(p):
            p = re.sub(r"\[.*\]", "", p)
            p = re.sub(r"\\n", " ", p)
            p = re.sub(r"\\t", " ", p)
            p = re.sub(r"\s+", " ", p)
            return p.strip()

        wp_limit = limit or 1000
        print("Generating and writing WP prompts...")

        tasks = []
        with open("data/wp/raw/train.wp_source", "r") as f:
            for num_lines_read, prompt in enumerate(f):
                if num_lines_read >= wp_limit:
                    break

                input_prompt = format_prompt(prompt)
                tasks.append((
                    f"data/wp/prompts/{num_lines_read + 1}.txt",
                    make_call(
                        f"Remove all the formatting in this prompt:\n\n{input_prompt}",
                        "gpt", args.gpt_model, args.debug,
                    ),
                ))

        run_parallel(tasks, args.workers, "wp prompts", stats)

    if args.wp_human:
        print("Formatting Human WP documents...")

        wp_limit = limit or 1000
        with open("data/wp/raw/train.wp_target", "r") as f:
            num_lines_read = 0

            pbar = tqdm.tqdm(total=wp_limit)
            for doc in f:
                if num_lines_read >= wp_limit:
                    break

                doc = doc.strip()
                tokens = doc.split(" ")

                replace = [
                    ["<newline>", "\n"],
                ]
                for r in replace:
                    tokens = [t.replace(r[0], r[1]) for t in tokens]

                detokenizer = TreebankWordDetokenizer()
                formatted_doc = detokenizer.detokenize(tokens)

                formatted_doc = "\n".join(
                    [i.strip() for i in formatted_doc.split("\n")]
                )
                formatted_doc = formatted_doc.replace("\n\n", "\n")
                formatted_doc = formatted_doc.replace("\n\n", "\n")

                formatted_doc = formatted_doc.replace(" .", ".")
                formatted_doc = formatted_doc.replace(" ’ ", "'")

                formatted_doc = formatted_doc.replace(" ”", '"')
                formatted_doc = formatted_doc.replace("“ ", '"')

                formatted_doc = html_replace(formatted_doc)

                with open(f"data/wp/human/{num_lines_read + 1}.txt", "w") as f:
                    f.write(formatted_doc)

                num_lines_read += 1
                pbar.update(1)

            pbar.close()

    wp_gpt_types = selected_gpt_types(args, "wp")
    if wp_gpt_types or args.wp_claude:
        wp_variants = []
        if wp_gpt_types:
            wp_variants.append((wp_gpt_types, "gpt", args.gpt_model))
        if args.wp_claude:
            wp_variants.append((prompt_types_claude, "claude", args.claude_model))

        print("Generating WP documents for:", ", ".join(t for types, _, _ in wp_variants for t in types))

        # exist_ok, and out here rather than inside the per-item loop: the
        # `if not exists: makedirs` pattern is a race between threads, and the
        # directory set is known before the first task anyway.
        for types, _, _ in wp_variants:
            for type in types:
                os.makedirs(f"data/wp/{out_dir_for_type(type, args)}", exist_ok=True)

        tasks = []
        for idx in range(1, (limit or 1000) + 1):
            with open(f"data/wp/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/wp/human/{idx}.txt", "r") as f:
                words = round_to_50(len(f.read().split(" ")))

            for types, mode, model in wp_variants:
                prompts = get_wp_prompts(words, prompt)

                for type in types:
                    out_path = f"data/wp/{out_dir_for_type(type, args)}/{idx}.txt"
                    # Resume check stays on the main thread, before submission:
                    # a worker doing exists-then-write would race with the write
                    # of whatever else is in flight for the same path.
                    if os.path.exists(out_path):
                        continue

                    tasks.append((
                        out_path,
                        make_call(
                            prompts[prompt_index_for_type(type)],
                            mode, model, args.debug,
                            post=lambda reply: reply.replace("\n\n", "\n"),
                        ),
                    ))

        run_parallel(tasks, args.workers, "wp documents", stats)

    if args.reuter_human:
        reuter_replace = ["--", "202-898-8312", "((", "($1=", "(A$", "Reuters Chicago"]

        authors = os.listdir("data/reuter/raw/C50train")
        print("Formatting Human Reuters documents...")

        author_files = []
        for author in authors:
            files = [
                f"data/reuter/raw/C50train/{author}/{i}"
                for i in os.listdir(f"data/reuter/raw/C50train/{author}")
            ] + [
                f"data/reuter/raw/C50test/{author}/{i}"
                for i in os.listdir(f"data/reuter/raw/C50test/{author}")
            ]
            author_files.extend(
                (author, n + 1, file)
                for n, file in enumerate(files[:REUTER_ARTICLES_PER_AUTHOR])
            )

        if limit is not None:
            author_files = author_files[:limit]

        for author, n, file in tqdm.tqdm(author_files):
            if not os.path.exists(f"data/reuter/human/{author}"):
                os.makedirs(f"data/reuter/human/{author}")

            with open(file, "r") as f:
                doc = f.read().strip()
                doc = doc.replace("\n\n", "\n")

                lines = doc.split("\n")
                if any([i in lines[-1] for i in reuter_replace]):
                    lines = lines[:-1]
                doc = "\n".join(lines)
                doc = html_replace(doc)

                with open(f"data/reuter/human/{author}/{n}.txt", "w") as f:
                    f.write(doc.strip())

    if args.reuter_prompts:
        print("Generating Reuters headlines...")

        authors = os.listdir("data/reuter/human")
        author_idx_pairs = [
            (author, idx)
            for author in authors
            for idx in range(1, REUTER_ARTICLES_PER_AUTHOR + 1)
        ]
        if limit is not None:
            author_idx_pairs = author_idx_pairs[:limit]

        def clean_headline(reply):
            return reply.replace("Headline: ", "").strip().strip("*").strip()

        tasks = []
        for author, idx in author_idx_pairs:
            os.makedirs(f"data/reuter/gpt/{author}/headlines", exist_ok=True)

            out_path = f"data/reuter/gpt/{author}/headlines/{idx}.txt"
            if os.path.exists(out_path):
                continue

            with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                doc = f.read().strip()

            tasks.append((
                out_path,
                make_call(
                    "Given the following news article, write a headline for it. "
                    "Respond with just the plain headline text, no markdown "
                    f"formatting or asterisks:\n\n{' '.join(doc.split(' ')[:500])}",
                    "gpt", args.gpt_model, args.debug,
                    post=clean_headline,
                ),
            ))

        run_parallel(tasks, args.workers, "reuter headlines", stats)

    reuter_gpt_types = selected_gpt_types(args, "reuter")
    if reuter_gpt_types or args.reuter_claude:
        reuter_variants = []
        if reuter_gpt_types:
            reuter_variants.append((reuter_gpt_types, "gpt", args.gpt_model))
        if args.reuter_claude:
            reuter_variants.append((prompt_types_claude, "claude", args.claude_model))

        print("Generating Reuters documents for:", ", ".join(t for types, _, _ in reuter_variants for t in types))

        authors = os.listdir("data/reuter/human")
        author_idx_pairs = [
            (author, idx)
            for author in authors
            for idx in range(1, REUTER_ARTICLES_PER_AUTHOR + 1)
        ]
        if limit is not None:
            author_idx_pairs = author_idx_pairs[:limit]

        tasks = []
        for author, idx in author_idx_pairs:
            with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                words = round_to_50(len(f.read().split(" ")))

            with open(f"data/reuter/gpt/{author}/headlines/{idx}.txt", "r") as f:
                headline = f.read().strip()

            for types, mode, model in reuter_variants:
                prompts = get_reuter_prompts(words, headline)

                for type in types:
                    # Note the headline read above comes from data/reuter/gpt/,
                    # not from here: headlines are a seeded input shared by
                    # every variant, so --out_name must not move them.
                    out_dir = out_dir_for_type(type, args)
                    os.makedirs(f"data/reuter/{out_dir}/{author}", exist_ok=True)

                    out_path = f"data/reuter/{out_dir}/{author}/{idx}.txt"
                    if os.path.exists(out_path):
                        continue

                    tasks.append((
                        out_path,
                        make_call(
                            prompts[prompt_index_for_type(type)],
                            mode, model, args.debug,
                            post=partial(strip_boilerplate,
                                         drop_preamble=args.strip_boilerplate),
                        ),
                    ))

        run_parallel(tasks, args.workers, "reuter documents", stats)

    if args.essay_human or args.essay_gpt:
        essay_dataset = load_dataset("qwedsacf/ivypanda-essays")

    if args.essay_human:
        print("Formatting Human Essay documents...")

        essay_limit = limit or 1000
        num_documents, idx = 0, 0
        pbar = tqdm.tqdm(total=essay_limit)

        while num_documents < essay_limit:
            essay = essay_dataset["train"][idx]
            essay = essay["TEXT"].strip()
            essay = essay[essay.index("\n") + 1 :]

            idx += 1

            if "table of contents" in essay.lower():
                continue

            essay = essay.replace("\n\n", "\n")
            lines = essay.split("\n")

            doc = []
            for line in lines:
                if any(
                    [
                        i in line.lower()
                        for i in [
                            "references",
                            "reference",
                            "work cited",
                            "works cited",
                            "bibliography",
                        ]
                    ]
                ):
                    break
                doc.append(line)
            doc = "\n".join(doc)

            with open(f"data/essay/human/{num_documents + 1}.txt", "w") as f:
                f.write(doc.strip())

            num_documents += 1
            pbar.update(1)

    if args.essay_prompts:
        print("Generating Essay prompts...")

        tasks = []
        for idx in range(1, (limit or 1000) + 1):
            with open(f"data/essay/human/{idx}.txt", "r") as f:
                doc = f.read().strip()

            tasks.append((
                f"data/essay/prompts/{idx}.txt",
                make_call(
                    "Given the following essay, write a prompt for it:\n\n"
                    f"{' '.join(doc.split(' ')[:500])}",
                    "gpt", args.gpt_model, args.debug,
                    post=lambda reply: reply.replace("Prompt: ", "").strip(),
                ),
            ))

        run_parallel(tasks, args.workers, "essay prompts", stats)

    essay_gpt_types = selected_gpt_types(args, "essay")
    if essay_gpt_types or args.essay_claude:
        essay_variants = []
        if essay_gpt_types:
            essay_variants.append((essay_gpt_types, "gpt", args.gpt_model))
        if args.essay_claude:
            essay_variants.append((prompt_types_claude, "claude", args.claude_model))

        print("Generating Essay documents for:", ", ".join(t for types, _, _ in essay_variants for t in types))

        for types, _, _ in essay_variants:
            for type in types:
                os.makedirs(f"data/essay/{out_dir_for_type(type, args)}", exist_ok=True)

        tasks = []
        for idx in range(1, (limit or 1000) + 1):
            with open(f"data/essay/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/essay/human/{idx}.txt", "r") as f:
                words = round_to_50(len(f.read().split(" ")))

            for types, mode, model in essay_variants:
                prompts = get_essay_prompts(words, prompt)

                for type in types:
                    out_path = f"data/essay/{out_dir_for_type(type, args)}/{idx}.txt"
                    if os.path.exists(out_path):
                        continue

                    tasks.append((
                        out_path,
                        make_call(
                            prompts[prompt_index_for_type(type)],
                            mode, model, args.debug,
                            post=partial(strip_boilerplate,
                                         drop_preamble=args.strip_boilerplate),
                        ),
                    ))

        run_parallel(tasks, args.workers, "essay documents", stats)

    if args.logprobs:
        datasets = [
            Dataset("normal", "data/wp/human"),
            Dataset("normal", "data/wp/gpt"),
            Dataset("author", "data/reuter/human"),
            Dataset("author", "data/reuter/gpt"),
            Dataset("normal", "data/essay/human"),
            Dataset("normal", "data/essay/gpt"),
        ]
        generate_logprobs(get_generate_dataset(*datasets))

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/other/ets"),
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
            Dataset("normal", "data/other/toefl91"),
            Dataset("normal", "data/other/undetectable"),
        ]

        generate_logprobs(get_generate_dataset(*other_datasets))

    if args.logprob_llama:
        print("Loading LLAMA...")
        # llama_7b = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(
        #     device
        # )
        llama_13b = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-AWQ").to(
            device
        )
        print("LLAMA Loaded")

        datasets = [
            Dataset("normal", "data/wp/human"),
            Dataset("normal", "data/wp/gpt"),
            Dataset("author", "data/reuter/human"),
            Dataset("author", "data/reuter/gpt"),
            Dataset("normal", "data/essay/human"),
            Dataset("normal", "data/essay/gpt"),
        ]
        generate_logprobs(
            get_generate_dataset(*datasets),
            # llama_7b_model=llama_7b,
            llama_13b_model=llama_13b,
        )

    if args.gen_perturb_char:

        def perturb_char_basic(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                peturb_type = np.random.choice(["swap", "delete", "insert"])
                if peturb_type == "swap":
                    idx = np.random.randint(len(doc) - 1)
                    doc = doc[:idx] + doc[idx + 1] + doc[idx] + doc[idx + 2 :]
                elif peturb_type == "delete" and len(doc) > 1:
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + doc[idx + 1 :]
                elif peturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = (
                        doc[:idx]
                        + np.random.choice(list(string.ascii_letters))
                        + doc[idx:]
                    )
            return doc

        def perturb_char_space(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                perturb_type = np.random.choice(["insert", "delete"])
                if perturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + " " + doc[idx:]
                elif perturb_type == "delete":
                    space_indices = [
                        idx for idx, c in enumerate(doc) if c == " " or c == "\n"
                    ]
                    if len(space_indices) > 0:
                        idx = np.random.choice(space_indices)
                        doc = doc[:idx] + doc[idx + 1 :]
            return doc

        def perturb_char_cap(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(doc))
                if doc[idx].isalpha():
                    if doc[idx].isupper():
                        doc = doc[:idx] + doc[idx].lower() + doc[idx + 1 :]
                    else:
                        doc = doc[:idx] + doc[idx].upper() + doc[idx + 1 :]
            return doc

        def perturb_word_adj(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(words) - 1)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            doc = " ".join(words)

            return doc

        def perturb_word_syn(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(words))
                word = words[idx]
                synonym = closest_synonym(word)
                if synonym:
                    words[idx] = synonym
            doc = " ".join(words)

            return doc

        perturb_char_word_fns = {
            "char_basic": perturb_char_basic,
            "char_space": perturb_char_space,
            "char_cap": perturb_char_cap,
            "word_adj": perturb_word_adj,
            "word_syn": perturb_word_syn,
        }

        if not os.path.exists("data/perturb"):
            os.makedirs("data/perturb")

        np.random.seed(args.seed)
        # Construct the test/train split. Seed of 0 ensures seriality across
        # all files performing the same split.
        indices = np.arange(6000)
        np.random.shuffle(indices)

        train, test = (
            indices[: math.floor(0.8 * len(indices))],
            indices[math.floor(0.8 * len(indices)) :],
        )

        # [4320 2006 5689 ... 4256 5807 4875] [5378 5980 5395 ... 1653 2607 2732]
        print("Train/Test Split:", train, test)
        files = generate_dataset_fn(lambda f: f, verbose=False)

        indices = np.arange(len(test))
        np.random.shuffle(indices)
        indices = indices[:200]

        labels = []
        for file in files[test][indices]:
            if "human" in file and "gpt" not in file:
                labels.append(0)
            elif "gpt" in file and "human" not in file:
                labels.append(1)
            else:
                raise ValueError("Invalid file name")

        with open("data/perturb/labels.txt", "w") as f:
            f.write("\n".join([str(i) for i in labels]))

        # Generate the perturbed documents
        num_perturb = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_char_word_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")

                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()

                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb_char:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_char_names
            for n in perturb_char_sizes
        ]

        generate_logprobs(get_generate_dataset(*perturb_datasets))

    if args.gen_perturb_sent:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using GPU")
        else:
            device = "cpu"
            print("Using CPU")

        tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
        model = PegasusForConditionalGeneration.from_pretrained(
            "tuner007/pegasus_paraphrase"
        ).to(device)

        def paraphrase(text):
            batch = tokenizer(
                [text], truncation=True, padding="longest", return_tensors="pt"
            ).to(device)
            translated = model.generate(**batch)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            return tgt_text[0]

        def perturb_sent_adj(doc, n=1):
            """
            Randomly swap n pairs of adjacent sentences in the document
            """
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 2:
                return (" ".join(doc)).strip()

            for _ in range(n):
                idx = np.random.randint(len(doc) - 1)
                doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]

            return (" ".join(doc)).strip()

        def perturb_sent_paraph(doc, n=1):
            """
            Randomly paraphrase n sentences in the document
            """
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 1:
                return (" ".join(doc)).strip()

            for _ in range(n):
                idx = np.random.randint(len(doc))
                doc[idx] = paraphrase(doc[idx])

            return (" ".join(doc)).strip()

        def perturb_para_adj(doc, n=1):
            """
            Randomly swap n pairs of adjacent paragraphs in the document
            """
            doc = doc.split("\n")
            if len(doc) < 2:
                return "\n".join(doc)

            for _ in range(n):
                idx = np.random.randint(len(doc) - 1)
                doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]
            return "\n".join(doc)

        def perturb_para_paraph(doc, n=1):
            """
            Randomly paraphrase n paragraphs in the document
            """
            doc = doc.split("\n")
            if len(doc) < 1:
                return "\n".join(doc)

            for _ in range(n):
                idx = np.random.randint(len(doc))
                doc[idx] = paraphrase(doc[idx])

            return "\n".join(doc)

        perturb_sent_fns = {
            "sent_adj": perturb_sent_adj,
            "sent_paraph": perturb_sent_paraph,
            "para_adj": perturb_para_adj,
            "para_paraph": perturb_para_paraph,
        }

        if not os.path.exists("data/perturb"):
            os.makedirs("data/perturb")

        np.random.seed(args.seed)
        # Construct the test/train split. Seed of 0 ensures seriality across
        # all files performing the same split.
        indices = np.arange(6000)
        np.random.shuffle(indices)

        train, test = (
            indices[: math.floor(0.8 * len(indices))],
            indices[math.floor(0.8 * len(indices)) :],
        )

        # [4320 2006 5689 ... 4256 5807 4875] [5378 5980 5395 ... 1653 2607 2732]
        print("Train/Test Split:", train, test)
        files = generate_dataset_fn(lambda f: f, verbose=False)

        indices = np.arange(len(test))
        np.random.shuffle(indices)
        indices = indices[:200]

        labels = []
        for file in files[test][indices]:
            if "human" in file and "gpt" not in file:
                labels.append(0)
            elif "gpt" in file and "human" not in file:
                labels.append(1)
            else:
                raise ValueError("Invalid file name")

        with open("data/perturb/labels.txt", "w") as f:
            f.write("\n".join([str(i) for i in labels]))

        # Generate the perturbed documents
        num_perturb = list(range(11))
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_sent_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")

                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()

                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb_sent:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_sent_names
            for n in perturb_sent_sizes
        ]

        generate_logprobs(get_generate_dataset(*perturb_datasets))

    print_and_log_summary(stats)

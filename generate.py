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
from datetime import datetime

# python generate.py --debug --limit 3           # mock LLM calls, first 3 items
# uv run python generate.py --limit 10 --wp_gpt           # real API calls, first 3 items only

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


_openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
        print(f"  [DEBUG] Skipping API call. Prompt: {messages[-1]['content'][:80]!r}...")
        return "[DEBUG] This is a mock LLM response used for testing."
    if mode == "gpt":
        response = openai_backoff(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    elif mode == "claude":
        response = claude_backoff(
            model=model,
            max_tokens=2048,
            messages=messages,
        )
        return response.content[0].text.strip()
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'gpt' or 'claude'")


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
        for path, error in stats["missed"]:
            lines.append(f"  {path}: {error}")

    summary = "\n".join(lines)
    print(summary)

    with open(log_path, "a") as f:
        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] generate.py run\n")
        f.write(summary)
        f.write("\n")


# Default number of articles to consider per Reuters author when --limit isn't given.
REUTER_ARTICLES_PER_AUTHOR = 20


def round_to_100(n):
    return int(round(n / 100.0)) * 100


def get_wp_prompts(words, prompt):
    return [
        f'Write a story in {words} words to the prompt "{prompt}."',
        f'You are an author, who is writing a story in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word story on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write a short story in response to the prompt "{prompt}."',
        f'Write a {words}-word story in the style of a beginner writer in response to the prompt "{prompt}."',
        f'Write a story with very short sentences in {words} words to the prompt "{prompt}."',
        f'Write a story in {words} words to the prompt "{prompt}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]


def get_reuter_prompts(words, headline):
    return [
        f'Write a news article in {words} words based on the headline "{headline}."',
        f'You are a news reporter, who is writing an article with the headline "{headline}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word news article based on the following headline: "{headline}." Could you please draft something for me?',
        f'Please help me write a New York Times article for the headline "{headline}."',
        f'Write a {words}-word news article in the style of a New York Times article based on the headline "{headline}."',
        f'Write a news article with very short sentences in {words} words based on the headline "{headline}."',
        f'Write a news article in {words} words based on the headline "{headline}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]


def get_essay_prompts(words, prompts):
    return [
        f'Write an essay in {words} words to the prompt "{prompt}."',
        f'You are a student, who is writing an essay in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word essay based on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write an essay in response to the prompt "{prompt}."',
        f"Write a {words}-word essay in the style of a high-school student  in response to the following prompt: {prompt}.",
        f'Write an essay with very short sentences in {words} words to the prompt "{prompt}."',
        f'Write an essay in {words} words to the prompt "{prompt}." Do not use any markdown formatting (no asterisks, headers, or bullet points) — write in plain prose only, as a human would.',
    ]


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
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: mock all LLM calls instead of making real API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap every loop to the first N items. Works with or without --debug, "
                             "so it can be used to test real API calls on a small sample.")

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

    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--logprob_other", action="store_true")
    parser.add_argument("--logprob_llama", action="store_true")

    parser.add_argument("--gen_perturb_char", action="store_true")
    parser.add_argument("--logprob_perturb_char", action="store_true")

    parser.add_argument("--gen_perturb_sent", action="store_true")
    parser.add_argument("--logprob_perturb_sent", action="store_true")

    args = parser.parse_args()

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
        with open("data/wp/raw/train.wp_source", "r") as f:
            num_lines_read = 0

            print("Generating and writing WP prompts...")

            pbar = tqdm.tqdm(total=wp_limit)
            for prompt in f:
                if num_lines_read >= wp_limit:
                    break

                input_prompt = format_prompt(prompt)
                out_path = f"data/wp/prompts/{num_lines_read + 1}.txt"

                try:
                    reply = call_llm(
                        messages=[{"role": "user", "content": f"Remove all the formatting in this prompt:\n\n{input_prompt}"}],
                        mode="gpt",
                        model=args.gpt_model,
                        debug=args.debug,
                    )

                    with open(out_path, "w") as out_f:
                        out_f.write(reply)
                    record_result(stats, out_path)
                except Exception as e:
                    record_result(stats, out_path, error=e)

                num_lines_read += 1
                pbar.update(1)

            pbar.close()

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

        for types, _, _ in wp_variants:
            for type in types:
                if not os.path.exists(f"data/wp/{type}"):
                    os.makedirs(f"data/wp/{type}")

        for idx in tqdm.tqdm(range(1, (limit or 1000) + 1)):
            with open(f"data/wp/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/wp/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))

            for types, mode, model in wp_variants:
                prompts = get_wp_prompts(words, prompt)

                for type in types:
                    out_path = f"data/wp/{type}/{idx}.txt"
                    if os.path.exists(out_path):
                        continue

                    variant_prompt = prompts[prompt_index_for_type(type)]

                    try:
                        reply = call_llm(
                            messages=[{"role": "user", "content": variant_prompt}],
                            mode=mode,
                            model=model,
                            debug=args.debug,
                        )
                        reply = reply.replace("\n\n", "\n")

                        with open(out_path, "w") as f:
                            f.write(reply)
                        record_result(stats, out_path)
                    except Exception as e:
                        record_result(stats, out_path, error=e)

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

        for author, idx in tqdm.tqdm(author_idx_pairs):
            if not os.path.exists(f"data/reuter/gpt/{author}/headlines"):
                os.makedirs(f"data/reuter/gpt/{author}/headlines")

            out_path = f"data/reuter/gpt/{author}/headlines/{idx}.txt"
            if os.path.exists(out_path):
                continue

            with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                doc = f.read().strip()

            try:
                reply = call_llm(
                    messages=[{"role": "user", "content": f"Given the following news article, write a headline for it. Respond with just the plain headline text, no markdown formatting or asterisks:\n\n{' '.join(doc.split(' ')[:500])}"}],
                    mode="gpt",
                    model=args.gpt_model,
                    debug=args.debug,
                )
                reply = reply.replace("Headline: ", "").strip()
                reply = reply.strip("*").strip()

                with open(out_path, "w") as f:
                    f.write(reply)
                record_result(stats, out_path)
            except Exception as e:
                record_result(stats, out_path, error=e)

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

        for author, idx in tqdm.tqdm(author_idx_pairs):
            with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))

            with open(f"data/reuter/gpt/{author}/headlines/{idx}.txt", "r") as f:
                headline = f.read().strip()

            for types, mode, model in reuter_variants:
                prompts = get_reuter_prompts(words, headline)

                for type in types:
                    if not os.path.exists(f"data/reuter/{type}/{author}"):
                        os.makedirs(f"data/reuter/{type}/{author}")

                    out_path = f"data/reuter/{type}/{author}/{idx}.txt"
                    if os.path.exists(out_path):
                        continue

                    variant_prompt = prompts[prompt_index_for_type(type)]

                    try:
                        reply = call_llm(
                            messages=[{"role": "user", "content": variant_prompt}],
                            mode=mode,
                            model=model,
                            debug=args.debug,
                        )
                        reply = reply.replace("\n\n", "\n")

                        lines = reply.split("\n")
                        if any([i in lines[0].lower() for i in ["sure", "certainly"]]):
                            reply = "\n".join(lines[1:])

                        lines = reply.split("\n")
                        if any([i in lines[0].lower() for i in ["title"]]):
                            reply = "\n".join(lines[1:])

                        with open(out_path, "w") as f:
                            f.write(reply)
                        record_result(stats, out_path)
                    except Exception as e:
                        record_result(stats, out_path, error=e)

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

        for idx in tqdm.tqdm(range(1, (limit or 1000) + 1)):
            out_path = f"data/essay/prompts/{idx}.txt"

            with open(f"data/essay/human/{idx}.txt", "r") as f:
                doc = f.read().strip()

            try:
                reply = call_llm(
                    messages=[{"role": "user", "content": f"Given the following essay, write a prompt for it:\n\n{' '.join(doc.split(' ')[:500])}"}],
                    mode="gpt",
                    model=args.gpt_model,
                    debug=args.debug,
                )
                reply = reply.replace("Prompt: ", "").strip()

                with open(out_path, "w") as f:
                    f.write(reply)
                record_result(stats, out_path)
            except Exception as e:
                record_result(stats, out_path, error=e)

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
                if not os.path.exists(f"data/essay/{type}"):
                    os.makedirs(f"data/essay/{type}")

        for idx in tqdm.tqdm(range(1, (limit or 1000) + 1)):
            with open(f"data/essay/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/essay/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))

            for types, mode, model in essay_variants:
                prompts = get_essay_prompts(words, prompt)

                for type in types:
                    out_path = f"data/essay/{type}/{idx}.txt"
                    if os.path.exists(out_path):
                        continue

                    variant_prompt = prompts[prompt_index_for_type(type)]

                    try:
                        reply = call_llm(
                            messages=[{"role": "user", "content": variant_prompt}],
                            mode=mode,
                            model=model,
                            debug=args.debug,
                        )
                        reply = reply.replace("\n\n", "\n")

                        lines = reply.split("\n")
                        if any([i in lines[0].lower() for i in ["sure", "certainly"]]):
                            reply = "\n".join(lines[1:])

                        lines = reply.split("\n")
                        if any([i in lines[0].lower() for i in ["title"]]):
                            reply = "\n".join(lines[1:])

                        with open(out_path, "w") as f:
                            f.write(reply)
                        record_result(stats, out_path)
                    except Exception as e:
                        record_result(stats, out_path, error=e)

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

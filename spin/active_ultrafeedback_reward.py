import asyncio
import importlib.util
import itertools
import math
import os
import time

import httpx
from openai import AsyncOpenAI

_CALL_COUNTER = itertools.count()
_STATS = {"started": 0, "finished": 0, "first_start": None}
_BATCH = {"idx": 0}

_LOG_DIR = "/iopsstor/scratch/cscs/dmelikidze/verl-training/logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_JOB_ID = os.environ.get("SLURM_JOB_ID", "nojob")
_LOG_FILE = open(os.path.join(_LOG_DIR, f"reward_job{_JOB_ID}_pid{os.getpid()}.log"), "a", buffering=1)

# Pre-computed scores indexed by position in batch, filled by the hook
_PRECOMPUTED: dict[int, dict] = {}


def _log(msg: str) -> None:
    line = f"[active_ultrafeedback_reward] {msg}\n"
    _LOG_FILE.write(line)
    _LOG_FILE.flush()


_PROMPTS_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/prompts.py"
_spec = importlib.util.spec_from_file_location("active_ultrafeedback_prompts", _PROMPTS_PATH)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

PREFERENCE_ANNOTATION_SYSTEM_PROMPT = _prompts.PREFERENCE_ANNOTATION_SYSTEM_PROMPT
INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT = _prompts.INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT
HONESTY_ANNOTATION_PROMPT = _prompts.HONESTY_ANNOTATION_PROMPT
TRUTHFULNESS_ANNOTATION_PROMPT = _prompts.TRUTHFULNESS_ANNOTATION_PROMPT
HELPFULNESS_ANNOTATION_PROMPT = _prompts.HELPFULNESS_ANNOTATION_PROMPT

PREFERENCE_ANNOTATION_AGAINST_JUDGE_OWN_RESPONSE_SYSTEM_PROMPT = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response based on a given criteria. You'll receive an input with three sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), <JUDGE_OWN_RESPONSE>...</JUDGE_OWN_RESPONSE> for the answer you once gave to the user prompt, and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response. 

Carefully read the provided input to understand the task, then assess how well the response fulfills the criteria requirements. If conversation history is present, ensure the response aligns with it; otherwise, evaluate based solely on the instruction. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

AGAINST_JUDGE_OWN_RESPONSE_ANNOTATION_PROMPT = """You will be evaluating the quality of an assistant's response to a user prompt.

Below is a scoring rubric from 1 to 5:
1. The response is very poor
2. The response is poor
3. The response is acceptable
4. The response is good
5. The response is excellent

Your output should only be an integer from 1 to 5. Do not output any additional text or explanations.

<USER_INPUT>{prompt}</USER_INPUT>

<JUDGE_OWN_RESPONSE>{judge_own_response}</JUDGE_OWN_RESPONSE>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

ASPECT2ANNOTATION_PROMPT = {
    # "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT,
    # "honesty": HONESTY_ANNOTATION_PROMPT,
    # "truthfulness": TRUTHFULNESS_ANNOTATION_PROMPT,
    # "helpfulness": HELPFULNESS_ANNOTATION_PROMPT,
    "judge_own_response": AGAINST_JUDGE_OWN_RESPONSE_ANNOTATION_PROMPT,
}

TARGET_TOKENS = ["1", "2", "3", "4", "5"]


def _format_prompt_input(prompt_data):
    if isinstance(prompt_data, str):
        return prompt_data.strip()
    if isinstance(prompt_data, list):
        if len(prompt_data) == 1:
            return prompt_data[0].get("content", "").strip()
        formatted = "### CONVERSATION HISTORY ###\n"
        for turn in prompt_data[:-1]:
            role = turn.get("role", "").upper()
            content = turn.get("content", "").strip()
            formatted += f"[{role}]: {content}\n\n"
        formatted += "### FINAL INSTRUCTION ###\n"
        formatted += prompt_data[-1].get("content", "").strip()
        return formatted
    return str(prompt_data)


def _expected_score(probs: dict[str, float]) -> float:
    return sum(int(tok) * probs.get(tok, 0.0) for tok in TARGET_TOKENS)


def _extract_probabilities(res) -> dict[str, float]:
    try:
        first_token_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        token_logprobs = {lp.token: lp.logprob for lp in first_token_logprobs}
        target_logprobs = {w: token_logprobs.get(w, -float("inf")) for w in TARGET_TOKENS}
        exp_values = [math.exp(lp) for lp in target_logprobs.values()]
        total = sum(exp_values)
        if total == 0:
            return {w: 0.0 for w in TARGET_TOKENS}
        return {k: float(v) / total for k, v in zip(target_logprobs.keys(), exp_values)}
    except Exception as e:
        _log(f"Failed to extract probabilities: {e}")
        return {w: 0.0 for w in TARGET_TOKENS}


async def _judge_aspect(
    client: AsyncOpenAI, model: str, aspect: str, formatted_input: str, completion: str, call_id: int,
    judge_own_response: str | None = None, 
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, float]:
    user_prompt = ASPECT2ANNOTATION_PROMPT[aspect].format(prompt=formatted_input, judge_own_response=judge_own_response, completion=completion)
    if judge_own_response is None:
        system_prompt = PREFERENCE_ANNOTATION_SYSTEM_PROMPT
    else:
        system_prompt = PREFERENCE_ANNOTATION_AGAINST_JUDGE_OWN_RESPONSE_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.monotonic()
    try:
        if semaphore is not None:
            async with semaphore:
                res = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=20,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
        else:
            res = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
    except Exception as e:
        _log(f"call#{call_id} aspect={aspect} FAILED after {time.monotonic() - t0:.2f}s: {type(e).__name__}: {e}")
        return {w: 0.0 for w in TARGET_TOKENS}
    probs = _extract_probabilities(res)
    return probs


async def _score_batch_async(items: list[dict]) -> list[dict]:
    """Score all items concurrently in a single event loop."""
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=100)
    semaphore = asyncio.Semaphore(512)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as http_client:
        client = AsyncOpenAI(
            base_url=items[0]["base_url"],
            api_key=items[0]["api_key"],
            http_client=http_client,
        )
        all_tasks = []
        task_index = []
        for idx, item in enumerate(items):
            for aspect in ASPECT2ANNOTATION_PROMPT:
                all_tasks.append(
                    _judge_aspect(
                        client, item["model"], aspect,
                        item["formatted_input"], item["completion"], item["call_id"],
                        semaphore=semaphore,
                    )
                )
                task_index.append((idx, aspect))

        all_results = await asyncio.gather(*all_tasks)

    payloads = []
    for _ in items:
        payloads.append({"aspect_scores": {}})
    for (idx, aspect), probs in zip(task_index, all_results):
        payloads[idx]["aspect_scores"][aspect] = _expected_score(probs)
    return payloads


def _precompute_batch(data, tokenizer, reward_fn_key) -> None:
    """Extract all prompts/responses from the batch, score them all concurrently,
    and store results in _PRECOMPUTED for compute_score to look up."""
    base_url = os.environ.get("JUDGE_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL")
    if not (base_url and api_key and model):
        _log("JUDGE env vars not set, skipping precompute")
        return

    items = []
    skipped_truncated = 0
    for i in range(len(data)):
        data_item = data[i]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        if data_source != "activeultrafeedback":
            continue

        skip_flag = data_item.non_tensor_batch.get("skip_reward_annotation", False)
        if skip_flag:
            _PRECOMPUTED[i] = {
                "score": 0.0,
                **{f"reward/{aspect}_score": 0.0 for aspect in ASPECT2ANNOTATION_PROMPT},
            }
            skipped_truncated += 1
            continue

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        formatted_input = _format_prompt_input(extra_info.get("prompt", prompt_str))

        call_id = next(_CALL_COUNTER)
        items.append({
            "idx": i,
            "call_id": call_id,
            "formatted_input": formatted_input,
            "completion": response_str,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        })

    if skipped_truncated > 0:
        _log(f"precompute: skipped {skipped_truncated} truncated responses (score=0.0)")

    if not items:
        return

    n_api = len(items) * len(ASPECT2ANNOTATION_PROMPT)
    _log(f"precompute: {len(items)} samples, {n_api} API calls concurrently")
    t0 = time.monotonic()
    payloads = asyncio.run(_score_batch_async(items))
    elapsed = time.monotonic() - t0
    _log(f"precompute done in {elapsed:.2f}s ({n_api} calls, {n_api / elapsed:.1f} calls/s)")

    for item, payload in zip(items, payloads):
        aspect_scores = payload["aspect_scores"]
        reward = sum(aspect_scores.values()) / len(aspect_scores) if aspect_scores else 0.0
        _PRECOMPUTED[item["idx"]] = {
            "score": float(reward),
            **{f"reward/{aspect}_score": float(s) for aspect, s in aspect_scores.items()},
        }



def _install_batch_hook() -> None:
    """Monkeypatch NaiveRewardManager.__call__ to precompute all rewards
    in parallel before the sequential loop calls compute_score."""
    try:
        from verl.workers.reward_manager.naive import NaiveRewardManager
    except Exception as e:
        _log(f"batch hook disabled (import failed): {e}")
        return
    if getattr(NaiveRewardManager.__call__, "_active_uf_hooked", False):
        return
    _orig_call = NaiveRewardManager.__call__

    def _patched_call(self, data, *args, **kwargs):
        _BATCH["idx"] += 1
        _PRECOMPUTED.clear()
        _PRECOMPUTED["_position_counter"] = 0
        t0 = time.monotonic()
        _log(f"batch#{_BATCH['idx']} start | size={len(data)}")
        _precompute_batch(data, self.tokenizer, self.reward_fn_key)
        try:
            return _orig_call(self, data, *args, **kwargs)
        finally:
            _log(
                f"batch#{_BATCH['idx']} end | "
                f"elapsed={time.monotonic() - t0:.1f}s"
            )

    _patched_call._active_uf_hooked = True
    NaiveRewardManager.__call__ = _patched_call
    _log("installed batch hook on NaiveRewardManager.__call__")


_install_batch_hook()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "activeultrafeedback":
        raise NotImplementedError(
            f"active_ultrafeedback_reward only supports data_source='activeultrafeedback', got {data_source!r}"
        )

    if extra_info is None or "prompt" not in extra_info:
        raise ValueError("active_ultrafeedback_reward requires extra_info['prompt']")

    # Look up precomputed result by position
    pos = _PRECOMPUTED.get("_position_counter", None)
    if pos is not None and pos in _PRECOMPUTED:
        result = _PRECOMPUTED[pos]
        _PRECOMPUTED["_position_counter"] = pos + 1
        _log(f"compute_score[{pos}] using precomputed reward={result['score']:.3f}")
        return result
    if pos is not None:
        _PRECOMPUTED["_position_counter"] = pos + 1

    # Fallback: sequential scoring
    _log(f"compute_score[{pos}] fallback to sequential")
    base_url = os.environ.get("JUDGE_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL")
    if not (base_url and api_key and model):
        raise RuntimeError("Set JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL environment variables")

    call_id = next(_CALL_COUNTER)
    formatted_input = _format_prompt_input(extra_info["prompt"])

    async def _score():
        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            tasks = [
                _judge_aspect(client, model, aspect, formatted_input, solution_str, call_id, extra_info.get("judge_own_response"))
                for aspect in ASPECT2ANNOTATION_PROMPT
            ]
            results = await asyncio.gather(*tasks)
        return dict(zip(ASPECT2ANNOTATION_PROMPT.keys(), results))

    aspect_probs = asyncio.run(_score())
    aspect_scores = {aspect: _expected_score(probs) for aspect, probs in aspect_probs.items()}
    reward = sum(aspect_scores.values()) / len(aspect_scores)
    return {
        "score": float(reward),
        **{f"reward/{aspect}_score": float(s) for aspect, s in aspect_scores.items()},
    }

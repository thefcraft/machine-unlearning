import torch
import numpy as np
from tqdm import tqdm


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 64,
) -> str:
    model.eval()
    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


@torch.inference_mode()
def perplexity(model, tokenizer, question: str, answer: str) -> float:
    """PPL ∈ [1, ∞) and 1 means perfect confidence."""
    model.eval()

    # Prompt only
    prompt_messages = [{"role": "user", "content": question}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prompt + answer
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    ).input_ids

    input_ids = torch.tensor(
        [full_ids],
        device=model.device,
    )

    labels = input_ids.clone()

    # Ignore prompt tokens
    labels[
        :, : len(prompt_ids)
    ] = -100  # NOTE: default ignore_index=-100 in nn.CrossEntropyLoss

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    loss = outputs.loss
    ppl = torch.exp(loss)

    return ppl.item()


def rouge_l(question: str, answer: str, beta: float = 1.2) -> float:
    ref = question.split()
    pred = answer.split()

    m, n = len(ref), len(pred)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if ref[i] == pred[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m

    if precision == 0 or recall == 0:
        return 0.0

    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)

@torch.inference_mode()
def evaluate_dataset(
    model,
    tokenizer,
    dataset,
):
    model.eval()

    perplexities = []
    rouges = []

    for sample in tqdm(dataset):
        question = sample["prompt"][0]["content"]
        answer = sample["completion"][0]["content"]

        ppl = perplexity(
            model,
            tokenizer,
            question,
            answer,
        )

        pred = generate(
            model,
            tokenizer,
            question,
            max_new_tokens=256,
        )

        rouge = rouge_l(
            question=answer,
            answer=pred,
        )

        perplexities.append(ppl)
        rouges.append(rouge)

    return {
        "ppl": float(np.mean(perplexities)),
        "rouge_l": float(np.mean(rouges)),
    }

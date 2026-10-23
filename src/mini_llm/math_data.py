import random

ADD = ["What is {a} + {b}?", "what is {a} + {b}", "what's {a}+{b}", "{a} + {b}", "How much is {a} plus {b}?"]
SUB = ["What is {a} - {b}?", "what is {a} - {b}", "what's {a}-{b}", "{a} - {b}", "How much is {a} minus {b}?"]
MUL = ["What is {a} * {b}?", "what is {a} x {b}", "what's {a}x{b}", "{a} x {b}", "How much is {a} times {b}?"]
DIV = [
    "What is {a} / {b}?", "what is {a} / {b}", "what's {a}/{b}", "{a} / {b}", "How much is {a} divided by {b}?",
]

TABLE_MAX = 12
WORDS = [
    "sun", "apple", "dance", "river", "cloud", "train", "night", "green", "smile", "bread",
    "ocean", "music", "dream", "horse", "stone", "light", "chair", "fruit", "quiet", "shore",
]

WIDTH_WEIGHTS = {1: 40, 2: 25, 3: 15, 4: 12, 5: 8}
MUL_WIDTH_WEIGHTS = {2: 35, 3: 30, 4: 25, 5: 10}


def number(rng, digits):
    low = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(low, 10 ** digits - 1)


def choose_width(rng, weights):
    total = sum(weights.values())
    r = rng.randint(1, total)
    for width in sorted(weights):
        r -= weights[width]
        if r <= 0:
            return width
    return max(weights)


def template(rng, base):
    return rng.choice([base[0], base[2], base[1], base[3], base[4]])


def fmt_answer(kind, a, b, result):
    sign = {"add": "+", "sub": "-", "mul": "x", "div": "/"}[kind]
    return f"{a} {sign} {b} = {result}."


def carried_answer(kind, a, b, result):
    if kind == "add":
        step = (b // 100) * 100
        if step <= 0 or step >= b:
            return fmt_answer(kind, a, b, result)
        mid = a + step
        return f"{a} + {b} = {a} + {step} = {mid}, then {mid} + {b - step} = {result}."
    step = (b // 100) * 100
    if step <= 0 or step >= b:
        return fmt_answer(kind, a, b, result)
    mid = a - step
    return f"{a} - {b} = {a} - {step} = {mid}, then {mid} - {b - step} = {result}."


def op_pair(rng):
    kind = rng.choices(("add", "sub", "mul"), weights=(5, 4, 2))[0]
    if kind in ("add", "sub"):
        width = choose_width(rng, WIDTH_WEIGHTS)
        a, b = number(rng, width), number(rng, width)
        if b > a:
            a, b = b, a
        result = a + b if kind == "add" else a - b
        answer = carried_answer(kind, a, b, result) if width >= 4 and rng.random() < 0.3 else fmt_answer(kind, a, b, result)
        return template(rng, ADD if kind == "add" else SUB).format(a=a, b=b), answer
    width = choose_width(rng, MUL_WIDTH_WEIGHTS)
    if width == 2:
        a, b = number(rng, 2), number(rng, 2)
    elif width == 3:
        a, b = number(rng, 3), rng.randint(1, 9)
    elif width == 4:
        a, b = number(rng, 4), rng.randint(1, 9)
    else:
        a, b = number(rng, 5), rng.randint(1, 9)
    return template(rng, MUL).format(a=a, b=b), fmt_answer("mul", a, b, a * b)


def table_pairs(seed=0):
    rng = random.Random(seed)
    pairs = []
    for a in range(1, TABLE_MAX + 1):
        for b in range(1, TABLE_MAX + 1):
            pairs.append((template(rng, MUL).format(a=a, b=b), fmt_answer("mul", a, b, a * b)))
    for p in range(2, TABLE_MAX + 1):
        for q in range(1, TABLE_MAX + 1):
            pairs.append((template(rng, DIV).format(a=p * q, b=p), f"{p * q} / {p} = {q}."))
    return pairs


def synthetic_pairs(count, seed=1, table_share=0.35):
    rng = random.Random(seed + 1)
    n_tables = int(count * table_share) // len(table_pairs(seed)) + 1
    pool = list(table_pairs(seed)) * n_tables
    ops = [op_pair(rng) for _ in range(count - min(count, int(count * table_share)))]
    pool = pool[: max(0, count - len(ops))] + ops
    rng.shuffle(pool)
    return pool[:count]


def count_pair(rng):
    n = rng.randint(3, 8)
    words = rng.sample(WORDS, n)
    return f'How many words are in this sentence? "{" ".join(words)}"', f"{n} words."


def sort_pair(rng):
    k = rng.randint(3, 5)
    nums = rng.sample(range(0, 10000), k)
    text = ", ".join(str(x) for x in nums)
    if rng.choices(("up", "down"), weights=(3, 1))[0] == "up":
        out = sorted(nums)
        suffix = "ascending"
    else:
        out = sorted(nums, reverse=True)
        suffix = "descending"
    return (
        f"Sort these numbers: {text}",
        ", ".join(str(x) for x in out) + f" ({suffix}).",
    )


def reverse_pair(rng):
    if rng.random() < 0.5:
        word = rng.choice(WORDS)
        return f'Say "{word}" backwards', f"{word[::-1]}."
    digits = str(number(rng, rng.randint(2, 5)))
    return f'Read "{digits}" backwards', f"{digits[::-1]}."


SKILLS = [(count_pair, 5), (sort_pair, 3), (reverse_pair, 2)]


def skill_pairs(count, seed=2):
    rng = random.Random(seed + 2)
    names = [f for f, w in SKILLS for _ in range(w)]
    return [rng.choice(names)(rng) for _ in range(count)]

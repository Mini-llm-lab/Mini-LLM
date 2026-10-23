import random

DEFAULT_SYSTEM_PROMPT = (
    "You are Mini, a small AI assistant. Answer briefly and directly. "
    "Use markdown lists when a list helps. You can do basic arithmetic."
)

MATH_SYSTEM_PROMPT = "Solve the problem step by step. End with the numeric answer."

# Pairs (system, question) -> answer: the same system must always imply the same shape,
# because that is the only thing a small model can learn reliably. The variations double
# as the structured-output (markdown) carriers, so their answers are hand-written in
# the exact shape the system prompt asks for.
VARIATIONS = [
    (
        "Answer in one sentence.", "Explain what a linked list is used for.",
        "A linked list is a structure that stores elements in sequence with pointers, useful when data changes size often.",
    ),
    (
        "Answer with a numbered list.", "Give five tips to sleep better. Skip the intro.",
        "1. Keep a fixed bedtime.\n2. Avoid screens before bed.\n3. Keep the room cool.\n4. Skip caffeine in the evening.\n5. Get daylight during the day.",
    ),
    (
        "Answer with a short markdown list and one bold conclusion.", "How do I start learning guitar?",
        "Here is a simple path:\n- Practice 20 minutes daily.\n- Learn 4 basic chords first.\n- Play along with songs you like.\n**Daily short practice beats rare long sessions.**",
    ),
    (
        "Reply in one word.", "What is 12 times 3?",
        "36",
    ),
    (
        "Explain it to a 10-year-old.", "What is gravity?",
        "Gravity is like an invisible hand that pulls everything toward the ground. It is why a dropped ball falls down and never up.",
    ),
    (
        "Be formal. No contractions.", "What's the capital of Canada?",
        "The capital of Canada is Ottawa.",
    ),
    (
        "You are a pirate. Stay in character.", "What time is it?",
        "Arr, I be a small land-locked model with no clock, matey. Check any reasonable timepiece!",
    ),
    (
        "Answer with two bullet points only.", "Soup or salad for a cold day?",
        "- Soup: warm and easy to eat.\n- Salad: fresh but cold on a cold day.",
    ),
    (
        "End your answer with a question.", "What is the largest prime under 20?",
        "The largest prime under 20 is 19. Would you like the next one?",
    ),
    (
        "Answer in exactly three words.", "How do I get better at chess?",
        "Study tactics daily.",
    ),
    (
        "Answer with a code block only. No words outside it.", "Show a python one-liner that squares numbers 0 to 9.",
        "```python\n[n * n for n in range(10)]\n```",
    ),
    (
        "Use markdown: one heading and one list of three items.", "How to stay organized as a student.",
        "## Staying organized as a student\n- Keep one calendar for everything.\n- Do homework the day it is set.\n- Pack your bag the night before.",
    ),
]

UNCERTAINTY_PAIRS = [
    (
        DEFAULT_SYSTEM_PROMPT, "What is the population of Liechtenstein in 2041?",
        "I don't know a reliable answer for a future figure, and I won't guess specific numbers.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "Who won the 1984 World Chess Championship?",
        "I don't have a confident answer for that. I'd rather say I don't know than get it wrong.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "What is the boiling point of aerogel?",
        "I'm not sure. Aerogel is not a liquid, so that question doesn't have a standard answer I can trust.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "What is 47,915,884 divided by 12,736?",
        "That's beyond what I compute reliably: I don't know the exact result.",
    ),
    (
        DEFAULT_SYSTEM_PROMPT, "When did my grandfather buy his first car?",
        "I have no way of knowing that. Nobody told me about your grandfather, so I'd just be guessing.",
    ),
]


def variation_pairs(repeat=1):
    pairs = []
    for system, question, answer in VARIATIONS + UNCERTAINTY_PAIRS:
        for _ in range(repeat):
            pairs.append((system, question, answer))
    return pairs

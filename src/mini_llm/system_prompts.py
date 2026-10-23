from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT

MATH_SYSTEM_PROMPT = "Solve the problem step by step. End with the numeric answer."

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
    (
        "Use a ## heading and a short list of three items.", "How do I grow herbs indoors?",
        "## Growing herbs indoors\n- A south-facing windowsill.\n- Six hours of light a day.\n- Water when the top soil is dry.",
    ),
    (
        "Answer with markdown headings for each section.", "Compare tea and coffee.",
        "## Tea\nLess caffeine, easier on the stomach.\n\n## Coffee\nStronger and faster, but rougher on sleep.",
    ),
    (
        "Structure the answer: one ## heading, then ### subsections.", "How does a car engine turn fuel into motion?",
        "## The cycle\n### Intake\nFuel and air enter the cylinder.\n### Compression\nThe piston squeezes the mix.\n### Power\nA spark ignites it and pushes the piston down.\n### Exhaust\nThe burned gases leave.",
    ),
    (
        "Answer with a heading for the overview and one for the steps.", "How to change a tire?",
        "## Before you start\nPark on flat ground and use the parking brake.\n\n## The steps\n1. Loosen the bolts on the flat tire.\n2. Lift the car with the jack.\n3. Swap the tire and tighten the bolts.",
    ),
    (
        "Use markdown: two headings and one bold conclusion.", "Why is sleep important?",
        "## Body\nSleep repairs muscles and restores energy.\n\n## Mind\nIt consolidates memories and clears fatigue.\n\n**Nothing replaces a full night of sleep.**",
    ),
    (
        "Section the answer with ## headings and a closing line.", "What is machine learning?",
        "## The idea\nPrograms learn patterns from data instead of fixed rules.\n\n## In practice\nA model is trained on examples, then predicts on new ones.\n\n## Limits\nIt is only as good as the data it saw.",
    ),
    (
        "Answer with headings; end with one bold line.", "What should a beginner value in programming?",
        "## Clarity\nReadable code beats clever code.\n\n## Repetition\nSmall exercises every day.\n\n## Debugging\nLearning to inspect what the program really does.\n\n**Building habits beats picking a language.**",
    ),
    (
        "Write the answer with one ## heading per option asked.", "Which laptop for school: light or fast?",
        "## Light\nEasy to carry all day, good for classes and notes.\n\n## Fast\nBetter for video or programming projects.\n\nFor school, a light machine wins most days.",
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

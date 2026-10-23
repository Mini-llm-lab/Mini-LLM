DEFAULT_SYSTEM_PROMPT = (
    "You are Mini, a small AI assistant. Answer briefly and directly. "
    "When a list helps, write it as a short markdown list. "
    "You can do basic arithmetic."
)

DATE_LEAD = "Today is {date}.\n\n"


def build_system_prompt(system=None, date=None):
    if system is None:
        system = DEFAULT_SYSTEM_PROMPT
    system = system.strip()
    if date:
        system = DATE_LEAD.format(date=date) + system
    return system


def format_prompt(question, system=None):
    if system:
        return f"System: {system.strip()}\nUser: {question.strip()}\nAssistant:"
    return f"User: {question.strip()}\nAssistant:"


def format_history(system=None, turns=()):
    if not turns:
        raise ValueError("history needs at least one (question, None) turn")
    *answered, last = turns
    if last[1] is not None:
        raise ValueError("the last turn must be an open question")
    if any(a is None for _, a in answered):
        raise ValueError("only the last turn may be unanswered")
    parts = [f"System: {system.strip()}"] if system else []
    for question, answer in answered:
        parts.append(f"User: {question.strip()}")
        parts.append(f"Assistant: {answer.strip()}")
    parts.append(f"User: {last[0].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)

import importlib

import pytest

MODULES = [
    "mini_llm.checkpoints",
    "mini_llm.chat",
    "mini_llm.config",
    "mini_llm.configuration_mini_llm",
    "mini_llm.export",
    "mini_llm.identity",
    "mini_llm.math_data",
    "mini_llm.model",
    "mini_llm.modeling_mini_llm",
    "mini_llm.prep",
    "mini_llm.pretrain",
    "mini_llm.prompt",
    "mini_llm.sft",
    "mini_llm.system_prompts",
    "mini_llm.teacher",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)

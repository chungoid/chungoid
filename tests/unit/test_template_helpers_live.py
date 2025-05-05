import pytest

from utils import template_helpers as th


def test_inject_variable():
    tpl = "Hello {{name}}!"
    out = th.inject_variable(tpl, "name", "World")
    assert out == "Hello World!"


def test_inject_multiple_variables():
    tpl = "A={{a}}, B={{b}}"
    out = th.inject_multiple_variables(tpl, {"a": "1", "b": "2"})
    assert out == "A=1, B=2"


def test_format_list_section_block():
    tpl = (
        "### ITEMS ###\nplaceholder\n### END_ITEMS ###"
    )
    formatted = th.format_list_section(tpl, "items", ["x", "y"], item_format="* {item}")
    assert "* x" in formatted and "* y" in formatted


def test_format_list_section_variable():
    tpl = "Deps:\n{{deps}}"
    deps = ["a", "b"]
    formatted = th.format_list_section(tpl, "deps", deps)
    assert "- a" in formatted and "- b" in formatted 
"""The `repr` filter used by IR templates must emit Python that runs."""

import math

import numpy as np
import pytest

from lobster.core.provenance.analysis_ir import AnalysisStep, _python_literal

PLAIN = [
    "a'b\"c\n{x}",
    0,
    -3,
    1.5,
    True,
    None,
    [1, "x", (2,)],
    {"k": [1, 2], 3: None},
    (),
    (1,),
    (1, 2),
    [],
    {},
]


@pytest.mark.parametrize("value", PLAIN, ids=repr)
def test_identical_to_repr_for_plain_values(value):
    assert _python_literal(value) == repr(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (float("inf"), math.inf),
        (float("-inf"), -math.inf),
        (np.float64(2.5), 2.5),
        (np.int64(3), 3),
        (np.bool_(True), True),
        (np.array([[1.0, np.inf]]), [[1.0, math.inf]]),
        ({"a": np.float32(0.5)}, {"a": 0.5}),
    ],
)
def test_non_repr_safe_values_evaluate_back(value, expected):
    assert eval(_python_literal(value)) == expected  # nosec B307


def test_nan_evaluates_to_nan():
    assert math.isnan(eval(_python_literal(float("nan"))))  # nosec B307


def test_template_with_inf_runs():
    ir = AnalysisStep(
        operation="t",
        tool_name="t",
        description="t",
        library="t",
        code_template="bound = {{ bound | repr }}",
        imports=[],
        parameters={"bound": float("inf")},
        parameter_schema={},
    )
    scope = {}
    exec(ir.render(), scope)  # nosec B102
    assert scope["bound"] == math.inf

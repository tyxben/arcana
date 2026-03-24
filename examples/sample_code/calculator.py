"""A simple calculator module."""

import json  # noqa: F401
import os  # noqa: F401

HISTORY = []


def add(a, b):
    return a + b


def divide(a, b):
    return a / b


def calculate(expression):
    try:
        result = eval(expression)
        HISTORY.append(result)
        return result
    except:  # noqa: E722  -- deliberate bad practice for code review demo
        return None


class Calculator:
    def __init__(self):
        self.memory = 0

    def store(self, value):
        self.memory = value

    def recall(self):
        return self.memory

    def batch_divide(self, numerator, denominators):
        results = []
        for d in denominators:
            results.append(numerator / d)
        return results

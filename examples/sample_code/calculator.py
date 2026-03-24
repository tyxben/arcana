"""A simple calculator module."""

import os  # noqa: F401
import json  # noqa: F401

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
    except:
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

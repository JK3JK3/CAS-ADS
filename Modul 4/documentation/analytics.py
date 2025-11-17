"""
analytics.py
============

A tiny utilities module showcasing **well-documented** Python code suitable for
automatic documentation generation with Sphinx.

This module includes:

- A `RollingStats` class for online mean/variance (Welford’s algorithm).
- A `TextCleaner` class for simple text normalization.
- A `Calculator` class with common math ops (with input validation).
- A `DataPipeline` orchestration helper that ties everything together.
- Typed, Google-style docstrings with examples that Sphinx's napoleon can parse.

Examples:
    >>> rs = RollingStats()
    >>> for x in [1, 2, 3]:
    ...     rs.update(x)
    >>> round(rs.mean, 3), round(rs.std, 3)
    (2.0, 1.0)

    >>> TextCleaner().clean("  Hello,   WORLD!!  ")
    'hello, world!!'

    >>> Calculator.add(2, 3)
    5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, List, Optional


class AnalyticsError(Exception):
    """Base exception for analytics-related errors."""


class ValidationError(AnalyticsError):
    """Raised when inputs fail validation."""


@dataclass
class RollingStats:
    """Online mean and variance using Welford’s algorithm.

    Tracks the number of observations, running mean, and sample variance
    as values are streamed in (no need to store all values in memory).

    Attributes:
        n (int): Count of observations.
        _mean (float): Internal running mean.
        _m2 (float): Sum of squares of differences from the current mean.

    Example:
        >>> rs = RollingStats()
        >>> rs.update(10)
        >>> rs.update(20)
        >>> round(rs.mean, 2)
        15.0
        >>> round(rs.variance, 2)  # sample variance with n=2
        50.0
    """

    n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float) -> None:
        """Update the running stats with a new value.

        Args:
            x (float): The new observation.

        Raises:
            ValidationError: If x is not a real finite number.
        """
        # Basic validation (expand as needed)
        if not isinstance(x, (int, float)):
            raise ValidationError("x must be int or float")

        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def bulk_update(self, values: Iterable[float]) -> None:
        """Update stats with a stream of values.

        Args:
            values (Iterable[float]): Values to ingest.

        Example:
            >>> rs = RollingStats()
            >>> rs.bulk_update([1, 2, 3, 4])
            >>> rs.n, rs.mean
            (4, 2.5)
        """
        for v in values:
            self.update(v)

    @property
    def mean(self) -> float:
        """float: The current running mean (0.0 if no observations)."""
        return self._mean if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        """float: Sample variance (0.0 if fewer than 2 observations)."""
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        """float: Sample standard deviation."""
        return sqrt(self.variance)


class TextCleaner:
    """Simple text normalization helpers.

    Methods here are intentionally straightforward to demo doc rendering.

    Example:
        >>> cleaner = TextCleaner()
        >>> cleaner.clean("  Hello,   WORLD!!  ")
        'hello, world!!'
    """

    def __init__(self, lowercase: bool = True, collapse_spaces: bool = True):
        """Initialize the cleaner.

        Args:
            lowercase (bool): If True, convert text to lowercase.
            collapse_spaces (bool): If True, normalize multiple spaces to one.
        """
        self.lowercase = lowercase
        self.collapse_spaces = collapse_spaces

    def clean(self, s: str) -> str:
        """Clean and normalize input text.

        Args:
            s (str): Raw input text.

        Returns:
            str: Cleaned text.

        Example:
            >>> TextCleaner(lowercase=False).clean("Hi   THERE")
            'Hi THERE'
        """
        if not isinstance(s, str):
            raise ValidationError("s must be a string")

        out = s.strip()
        if self.collapse_spaces:
            out = " ".join(out.split())
        if self.lowercase:
            out = out.lower()
        return out


class Calculator:
    """Typed math helpers with basic validation."""

    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a (float): First addend.
            b (float): Second addend.

        Returns:
            float: Sum.

        Example:
            >>> Calculator.add(1.5, 2.5)
            4.0
        """
        Calculator._validate_numbers(a, b)
        return a + b

    @staticmethod
    def mul(a: float, b: float) -> float:
        """Multiply two numbers.

        Args:
            a (float): First factor.
            b (float): Second factor.

        Returns:
            float: Product.

        Example:
            >>> Calculator.mul(3, 4)
            12
        """
        Calculator._validate_numbers(a, b)
        return a * b

    @staticmethod
    def mean(values: Iterable[float]) -> float:
        """Compute arithmetic mean.

        Args:
            values (Iterable[float]): Numeric values.

        Returns:
            float: Mean (0.0 if empty).

        Example:
            >>> Calculator.mean([2, 4, 6])
            4.0
        """
        total = 0.0
        count = 0
        for v in values:
            Calculator._validate_numbers(v)
            total += v
            count += 1
        return total / count if count else 0.0

    @staticmethod
    def _validate_numbers(*nums: float) -> None:
        """Internal: validate numeric inputs."""
        for n in nums:
            if not isinstance(n, (int, float)):
                raise ValidationError("all inputs must be int or float")


@dataclass
class DataPipeline:
    """A tiny orchestration helper for analytics.

    Demonstrates how multiple components may be combined into a simple flow.

    Attributes:
        cleaner (TextCleaner): Text normalization component.
        stats (RollingStats): Streaming stats component.
        cache (List[float]): Optionally accumulated numeric features.

    Example:
        >>> p = DataPipeline()
        >>> p.ingest_texts(["  Hello  ", "WORLD   "])
        >>> p.cleaner.clean("  EXTRA   Text ")
        'extra text'
        >>> p.ingest_numbers([10, 20, 30])
        >>> round(p.stats.mean, 2)
        20.0
    """

    cleaner: TextCleaner = field(default_factory=TextCleaner)
    stats: RollingStats = field(default_factory=RollingStats)
    cache: List[float] = field(default_factory=list)

    def ingest_texts(self, texts: Iterable[str]) -> List[str]:
        """Clean a stream of texts and return cleaned copies.

        Args:
            texts (Iterable[str]): Raw strings.

        Returns:
            List[str]: Cleaned strings.

        Raises:
            ValidationError: If any item is not a string.
        """
        cleaned: List[str] = []
        for t in texts:
            cleaned.append(self.cleaner.clean(t))
        return cleaned

    def ingest_numbers(self, xs: Iterable[float]) -> None:
        """Update rolling stats and keep an optional cache.

        Args:
            xs (Iterable[float]): Numbers to ingest.

        Raises:
            ValidationError: If any item is not numeric.
        """
        for x in xs:
            self.stats.update(x)
            self.cache.append(float(x))

    def reset(self) -> None:
        """Reset the pipeline state (stats + cache)."""
        self.stats = RollingStats()
        self.cache.clear()


def demo() -> None:
    """Quick demo printing pipeline results.

    This function is only here to demonstrate a “script-like” entrypoint that
    still has good docstrings for auto-doc generation.

    Example:
        >>> demo()  # doctest: +SKIP
    """
    p = DataPipeline()
    p.ingest_texts(["  Hello  ", "WORLD   "])
    p.ingest_numbers([1, 2, 3, 4, 5])
    print(f"n={p.stats.n} mean={p.stats.mean:.2f} std={p.stats.std:.2f}")


if __name__ == "__main__":
    demo()

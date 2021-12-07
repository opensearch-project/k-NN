# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Provides decorators to profile functions.

The decorators work by adding a `measureable` (time, memory, etc) field to a
dictionary returned by the wrapped function. So the wrapped functions must
return a dictionary in order to be profiled.
"""
import functools
import time
from typing import Callable


class TimerStoppedWithoutStartingError(Exception):
    """Error raised when Timer is stopped without having been started."""

    def __init__(self):
        super().__init__()
        self.message = 'Timer must call start() before calling end().'


class _Timer():
    """Timer class for timing.

    Methods:
        start: Starts the timer.
        end: Stops the timer and returns the time elapsed since start.

    Raises:
        TimerStoppedWithoutStartingError: Timer must start before ending.
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()

    def end(self) -> float:
        """Stops the timer.

        Returns:
            The time elapsed in milliseconds.
        """
        # ensure timer has started before ending
        if self.start_time is None:
            raise TimerStoppedWithoutStartingError()

        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.start_time = None
        return elapsed


def took(f: Callable):
    """Profiles a functions execution time.

    Args:
        f: Function to profile.

    Returns:
        A function that wraps the passed in function and adds a time took field
        to the return value.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        timer = _Timer()
        timer.start()
        result = f(*args, **kwargs)
        time_took = timer.end()

        # if result already has a `took` field, don't modify the result
        if isinstance(result, dict) and 'took' in result:
            return result
        # `result` may not be a dictionary, so it may not be unpackable
        elif isinstance(result, dict):
            return {**result, 'took': time_took}
        return {'took': time_took}

    return wrapper

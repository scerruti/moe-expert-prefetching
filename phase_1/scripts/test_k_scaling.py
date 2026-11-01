#!/usr/bin/env python3
"""The buffer calculation behind the k-scaling experiment."""

import numpy as np

from k_scaling import needed_buffer, ranks, smallest_m


def test_needed_buffer_is_the_extra_depth_required():
    a = ranks(np.array([[0.5, 0.3, 0.15, 0.05]]))
    swapped = ranks(np.array([[0.5, 0.15, 0.3, 0.05]]))  # ranks 2 and 3 swap
    last = ranks(np.array([[0.05, 0.3, 0.15, 0.5]]))     # top and bottom swap

    assert needed_buffer(a, a, 2)[0] == 0
    assert needed_buffer(a, swapped, 2)[0] == 1
    assert needed_buffer(a, swapped, 1)[0] == 0
    assert needed_buffer(a, last, 1)[0] == 3


def test_smallest_m_uses_changed_decisions_only():
    histogram = np.array([1000, 90, 9, 1])  # m=0 unchanged; m=1, 2, 3 needed
    assert smallest_m(histogram, 0.90) == 1
    assert smallest_m(histogram, 0.95) == 2
    assert smallest_m(histogram, 0.995) == 3

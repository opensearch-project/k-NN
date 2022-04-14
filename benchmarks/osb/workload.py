# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

# This code needs to be included at the top of every workload.py file.
# OpenSearch Benchmarks is not able to find other helper files unless the path
# is updated.
import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from extensions.registry import register as custom_register


def register(registry):
    custom_register(registry)

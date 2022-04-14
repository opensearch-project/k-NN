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
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from helpers.registry import register as custom_register

#TODO:
# 1. Add support for training
#   a. Custom runner to train model
#   b. Custom runner to delete model
#   c. Enhance bulk for setting limit on training data
#   d. Refactor to use challenges instead of multiple workloads

def register(registry):
    custom_register(registry)

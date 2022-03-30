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
# 3. Add support for training
# 4. Add validation (Similar to steps)
# 5. Add different requirements.txt file
# 6. Add descriptive documentation
# ------------------------ PR ----------------------------
# 7. Add query workloads
# 8. Add recall computation workloads
# 9. Add checkstyles for python


def register(registry):
    custom_register(registry)

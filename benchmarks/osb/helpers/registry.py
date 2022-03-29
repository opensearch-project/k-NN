# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from helpers.param_sources import register as param_sources_register
from helpers.runners import register as runners_register


def register(registry):
    param_sources_register(registry)
    runners_register(registry)

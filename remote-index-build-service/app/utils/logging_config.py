#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0
import logging

def configure_logging(log_level):
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

def version_range(min_version='1.0', max_version='3.1'):
    """Decorator to specify version range for tests"""
    def decorator(func):
        func._min_version = min_version
        func._max_version = max_version
        return func
    return decorator

def normalize_version(version_string):
    """Normalize version string by removing -SNAPSHOT and extra parts"""
    # Remove -SNAPSHOT and other suffixes
    version = version_string.split('-')[0]
    # Keep only major.minor (e.g., 3.1.0 -> 3.1)
    parts = version.split('.')
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else version
#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

import pytest
import os
from opensearchpy import OpenSearch
import uuid
from packaging import version
from version_utils import normalize_version
from faker import Faker

# Global variable to store cluster version
cluster_version = '1.0'

@pytest.fixture(scope="session")
def opensearch_client():
    """Create OpenSearch client for external cluster"""
    client = create_client()
    info = client.info()
    num_nodes = int(os.getenv('OPENSEARCH_NODES', '1'))
    print(f"Connected to OpenSearch {info['version']['number']} ({num_nodes} nodes)")
    yield client
    client.close()

@pytest.fixture
def faker():
    """Seeded faker for reproducible test data"""
    fake = Faker()
    fake.seed_instance(42)
    return fake

@pytest.fixture
def test_index():
    """Generate unique test index name"""
    return f"test_knn_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def index_manager(opensearch_client):
    """Manages index cleanup after tests"""
    indices_to_delete = []

    def register_index(index_name):
        indices_to_delete.append(index_name)
        return index_name

    yield register_index

    # Cleanup all registered indices
    for index in indices_to_delete:
        opensearch_client.indices.delete(index=index, ignore=404)


def pytest_collection_modifyitems(config, items):
    """Filter tests based on cluster version"""
    # Get cluster version during collection
    try:
        client = create_client()
        info = client.info()
        detected_version = normalize_version(info['version']['number'])
        client.close()
        print(f"Detected cluster version: {detected_version}")
    except:
        detected_version = '1.0'
        print("Could not detect cluster version, using default 1.0")
    
    for item in items:
        min_ver = getattr(item.function, '_min_version', '1.0')
        max_ver = getattr(item.function, '_max_version', '3.1')

        if not (version.parse(min_ver) <= version.parse(detected_version) <= version.parse(max_ver)):
            item.add_marker(pytest.mark.skip(f"Version {detected_version} not in range {min_ver}-{max_ver}"))

def create_client():
    """Create OpenSearch client for version detection"""
    host = os.getenv('OPENSEARCH_HOST', 'localhost')
    port = int(os.getenv('OPENSEARCH_PORT', '9200'))
    username = os.getenv('OPENSEARCH_USERNAME', '')
    password = os.getenv('OPENSEARCH_PASSWORD', '')
    use_ssl = os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true'
    
    return OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=(username, password),
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False
    )


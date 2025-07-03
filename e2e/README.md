# OpenSearch k-NN E2E Tests

Python-based end-to-end tests for the OpenSearch k-NN plugin against external clusters.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Using Gradle (Recommended):
```bash
# Run all tests against localhost:9200
./gradlew :e2e:runTests

# Run against custom cluster
./gradlew :e2e:runTests -Dopensearch.host=my-cluster.com -Dopensearch.port=9200

# Run against multi-node cluster
./gradlew :e2e:runTests -Dopensearch.nodes=3

# Run specific test pattern
./gradlew :e2e:runTests -Dtest.pattern="test_knn_basic"

# Run with custom credentials
./gradlew :e2e:runTests -Dopensearch.username=myuser -Dopensearch.password=mypass
```

### Using Python directly:
```bash
# All tests
python run_tests.py

# Specific test pattern
python run_tests.py -k "test_knn_basic"

# With pytest directly
pytest -v
```

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `tests/test_knn_basic.py` - Basic k-NN functionality tests

## Configuration

### Gradle Task Parameters
- `-Dopensearch.host` - Cluster host (default: localhost)
- `-Dopensearch.port` - Cluster port (default: 9200)
- `-Dopensearch.username` - Username (default: admin)
- `-Dopensearch.password` - Password (default: admin)
- `-Dopensearch.ssl` - Use SSL (default: true)
- `-Dopensearch.nodes` - Number of nodes (default: 1)
- `-Dtest.pattern` - Test pattern to match

### Environment Variables
- `OPENSEARCH_HOST` - Cluster host (default: localhost)
- `OPENSEARCH_PORT` - Cluster port (default: 9200)
- `OPENSEARCH_USERNAME` - Username (default: admin)
- `OPENSEARCH_PASSWORD` - Password (default: admin)
- `OPENSEARCH_USE_SSL` - Use SSL (default: true)
- `OPENSEARCH_NODES` - Number of nodes (default: 1)
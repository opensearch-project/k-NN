#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from opensearchpy import OpenSearch

from version_utils import version_range


class TestKNNBasic:
    """Basic k-NN functionality tests"""

    @version_range(min_version='3.0')
    def test_knn_index_creation(self, opensearch_client: OpenSearch, test_index, index_manager):

        index_manager(test_index)

        """Test creating k-NN index"""
        body = {
            "settings": {
              "knn": True,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 3,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "faiss"
                        }
                    }
                }
            }
        }
        
        response = opensearch_client.indices.create(index=test_index, body=body)
        assert response["acknowledged"]

    @version_range(min_version='3.1')
    def test_knn_document_indexing(self, opensearch_client: OpenSearch, test_index, index_manager, faker):

        index_manager(test_index)

        """Test indexing documents with vectors"""
        body = {
            "settings": {
              "knn": True,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 3
                    }
                }
            }
        }
        
        opensearch_client.indices.create(index=test_index, body=body)
        
        # Index test documents
        docs = [
            {"vector": [1.0, 2.0, 3.0], "title": faker.word()},
            {"vector": [2.0, 3.0, 4.0], "title": faker.word()},
            {"vector": [3.0, 4.0, 5.0], "title": faker.word()}
        ]
        
        for i, doc in enumerate(docs):
            opensearch_client.index(index=test_index, id=i, body=doc)
        
        opensearch_client.indices.refresh(index=test_index)
        
        # Verify documents
        count = opensearch_client.count(index=test_index)
        assert count["count"] == 3
    
    def test_knn_search(self, opensearch_client: OpenSearch, test_index, index_manager, faker):

        index_manager(test_index)

        """Test k-NN search functionality"""
        body = {
            "settings": {
              "knn": True,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 3
                    }
                }
            }
        }
        
        opensearch_client.indices.create(index=test_index, body=body)
        
        # Index test documents
        docs = [
            {"vector": [1.0, 1.0, 1.0], "title": "doc1"},
            {"vector": [2.0, 2.0, 2.0], "title": "doc2"},
            {"vector": [10.0, 10.0, 10.0], "title": "doc3"}
        ]
        
        for i, doc in enumerate(docs):
            opensearch_client.index(index=test_index, id=i, body=doc)
        
        opensearch_client.indices.refresh(index=test_index)

        # Perform k-NN search
        query = {
            "size": 2,
            "query": {
                "knn": {
                    "vector": {
                        "vector": [1.5, 1.5, 1.5],
                        "k": 2
                    }
                }
            }
        }
        
        response = opensearch_client.search(index=test_index, body=query)
        
        assert len(response["hits"]["hits"]) == 2
        # Verify we got results
        assert all("title" in hit["_source"] for hit in response["hits"]["hits"])

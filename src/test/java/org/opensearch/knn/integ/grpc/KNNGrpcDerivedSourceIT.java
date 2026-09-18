/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ.grpc;

import com.google.protobuf.ByteString;
import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.protobufs.BulkResponse;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.grpc.ManagedChannel;

/**
 * gRPC transport integration tests for bulk ingestion into a derived-source {@code knn_vector} index.
 *
 * <p>Baseline: a JSON-encoded document ingests over gRPC. Regression guard: a CBOR-encoded document (the
 * shape produced by the CBOR-over-gRPC bulk optimization) ingests into the same derived-source index. The
 * CBOR case is the end-to-end guard for the k-NN derived-source fix — before that fix the stored-fields
 * writer hardcoded a JSON parser and failed CBOR ingestion with "Failed to parse content to map".
 *
 * <p>Each test asserts the gRPC bulk response is error-free, then reads the document back over REST and
 * verifies the derived vector field is reconstructed, both before and after a force merge (merge re-invokes
 * the derived-source stored-fields writer).
 */
public class KNNGrpcDerivedSourceIT extends KNNGrpcIntegTestCase {

    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 3;
    private static final List<Float> VECTOR = List.of(1.5f, 2.5f, 3.5f);

    @SneakyThrows
    public void testGrpcBulkJsonIntoDerivedSourceIndex() {
        assertGrpcBulkRoundTrip(XContentType.JSON);
    }

    @SneakyThrows
    public void testGrpcBulkCborIntoDerivedSourceIndex() {
        assertGrpcBulkRoundTrip(XContentType.CBOR);
    }

    @SneakyThrows
    private void assertGrpcBulkRoundTrip(MediaType mediaType) {
        String indexName = getTestName().toLowerCase(java.util.Locale.ROOT);
        createDerivedSourceKnnIndex(indexName);

        Map<String, Object> source = new LinkedHashMap<>();
        source.put(FIELD_NAME, VECTOR);
        source.put("text_field", "hello");

        ManagedChannel channel = newGrpcChannel();
        try {
            ByteString body = encodeSource(source, mediaType);
            BulkResponse response = grpcIndexDocument(channel, indexName, "1", body);

            assertFalse(
                "gRPC bulk of a " + mediaType + " document into a derived-source knn_vector index must not error",
                response.getErrors()
            );
            assertEquals("expected exactly one bulk item", 1, response.getItemsCount());
            // A successful item must carry no error. (The proto status field is not a reliable HTTP 201
            // mirror across versions; the authoritative success signals are getErrors()==false, an absent
            // item error, and the read-back below.)
            org.opensearch.protobufs.ResponseItem item = response.getItems(0).getIndex();
            assertFalse("bulk item must not carry an error: " + item.getError().getReason(), item.hasError());
        } finally {
            shutdownChannel(channel);
        }

        // Read back over REST: derived source must reconstruct the vector from the HNSW index.
        assertVectorReconstructed(indexName);

        // Force merge re-invokes the derived-source stored-fields writer's mask path.
        forceMergeKnnIndex(indexName, 1);
        assertVectorReconstructed(indexName);
    }

    @SneakyThrows
    private void createDerivedSourceKnnIndex(String indexName) {
        Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("index.number_of_shards", 1)
            .put("index.number_of_replicas", 0)
            .put("index.knn.derived_source.enabled", true)
            .build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "faiss")
            .field("space_type", "l2")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, settings, mapping);
    }

    @SneakyThrows
    @SuppressWarnings("unchecked")
    private void assertVectorReconstructed(String indexName) {
        Map<String, Object> source = getKnnDoc(indexName, "1");
        assertEquals("text field must be preserved", "hello", source.get("text_field"));
        assertTrue("derived source must reconstruct the vector field on read", source.containsKey(FIELD_NAME));
        List<Number> actual = (List<Number>) source.get(FIELD_NAME);
        assertNotNull("vector must not be masked/null after read-back", actual);
        assertEquals(VECTOR.size(), actual.size());
        for (int i = 0; i < VECTOR.size(); i++) {
            assertEquals(VECTOR.get(i), actual.get(i).floatValue(), 0.0001f);
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.HttpHost;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.client.RestClient;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.junit.Before;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;

/**
 * Multi-node integration tests for dynamic knn_vector mapping. Runs under {@code integTestMultiNode}
 * (2-node cluster); self-skips under single-node {@code integTest}.
 */
@Log4j2
public class DynamicVectorMappingMultiNodeIT extends KNNRestTestCase {

    @Before
    public void enableDynamicMappingFeature() throws Exception {
        // Cluster setting is off by default (see KNNSettings.KNN_DYNAMIC_MAPPING_ENABLED_SETTING);
        // this test exercises the feature, so opt in here.
        updateClusterSettings(KNNSettings.KNN_DYNAMIC_MAPPING_ENABLED, true);
    }

    private static String numericArray(int n) {
        return "[" + IntStream.range(0, n).mapToObj(i -> "0.1").collect(Collectors.joining(",")) + "]";
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> fieldMapping(String index, String field) throws IOException, ParseException {
        Request request = new Request("GET", "/" + index + "/_mapping");
        Response response = client().performRequest(request);
        Map<String, Object> body = createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            EntityUtils.toString(response.getEntity())
        ).map();
        Map<String, Object> indexEntry = (Map<String, Object>) body.get(index);
        if (indexEntry == null) return null;
        Map<String, Object> mappings = (Map<String, Object>) indexEntry.get("mappings");
        if (mappings == null) return null;
        Map<String, Object> properties = (Map<String, Object>) mappings.get("properties");
        if (properties == null) return null;
        return (Map<String, Object>) properties.get(field);
    }

    /**
     * Fires two competing dynamic-inference ingests, one per node, with different dimensions and asserts:
     * <ul>
     *   <li>exactly one doc is ingested (the other fails on the mapping conflict);</li>
     *   <li>the field resolves to {@code knn_vector} with a single locked dimension (no split-brain);</li>
     *   <li>the surviving doc's dimension matches the locked one.</li>
     * </ul>
     * The two per-node {@link RestClient}s route the requests to different coordinators, so the two
     * dynamic-mapping updates arrive at the cluster manager independently — the actual race Kunal raised.
     */
    public void testConcurrentIngestOnDifferentNodesLocksToOneDimension() throws Exception {
        // getClusterHosts() lists each node twice (IPv6 + IPv4 endpoints on adjacent ports), so we cannot
        // pick hosts.get(0) and hosts.get(1) blindly — they may resolve to the same node. Ask the cluster
        // for the authoritative one-per-node HTTP address list via _cat/nodes.
        List<HttpHost> nodeHosts = getPerNodeHttpHosts();
        if (nodeHosts.size() < 2) {
            log.warn("Skipping testConcurrentIngestOnDifferentNodesLocksToOneDimension: need >= 2 nodes, found {}", nodeHosts.size());
            return;
        }

        final int dimA = 128;
        final int dimB = 256;
        final String index = "dv_race_2node";
        createIndex(index, getKNNDefaultIndexSettings());

        try (
            RestClient nodeAClient = buildClient(restClientSettings(), new HttpHost[] { nodeHosts.get(0) });
            RestClient nodeBClient = buildClient(restClientSettings(), new HttpHost[] { nodeHosts.get(1) })
        ) {
            ExecutorService pool = Executors.newFixedThreadPool(2);
            CountDownLatch startGate = new CountDownLatch(1);
            Future<IngestResult> fA = pool.submit(() -> ingestOnNode(nodeAClient, dimA, index, startGate));
            Future<IngestResult> fB = pool.submit(() -> ingestOnNode(nodeBClient, dimB, index, startGate));
            startGate.countDown();

            IngestResult rA = fA.get(60, TimeUnit.SECONDS);
            IngestResult rB = fB.get(60, TimeUnit.SECONDS);
            pool.shutdown();
            assertTrue("executor did not terminate", pool.awaitTermination(30, TimeUnit.SECONDS));

            // Exactly one ingest must succeed — the other must be rejected on mapping conflict.
            assertTrue("both ingests succeeded, race resolution incorrect: A=" + rA.status() + " B=" + rB.status(), !(rA.ok() && rB.ok()));
            assertTrue("no ingest succeeded, race resolution incorrect: A=" + rA.status() + " B=" + rB.status(), rA.ok() || rB.ok());

            // The losing side must have failed specifically due to a dimension-mapping conflict, not some other error.
            IngestResult loser = rA.ok() ? rB : rA;
            assertEquals("losing side must return 400, got " + loser.status(), 400, loser.status());
            assertTrue(
                "losing side error body must reflect the dimension mapping conflict, got: " + loser.errorBody(),
                loser.errorBody().contains("Cannot update parameter [dimension]")
            );

            // The surviving dimension is the one locked into the mapping.
            int expectedLocked = rA.ok() ? dimA : dimB;
            refreshIndex(index);
            Map<String, Object> mapping = fieldMapping(index, "emb");
            assertNotNull("emb field mapping missing after race", mapping);
            assertEquals(KNN_VECTOR, mapping.get(TYPE));
            int lockedDim = ((Number) mapping.get(DIMENSION)).intValue();
            assertEquals("locked dimension must match the surviving ingest", expectedLocked, lockedDim);
        } finally {
            deleteKNNIndex(index);
        }
    }

    /** Returns one {@link HttpHost} per node, from {@code _cat/nodes} — avoids the IPv4/IPv6 duplication in {@code getClusterHosts()}. */
    private List<HttpHost> getPerNodeHttpHosts() throws IOException {
        Response resp = client().performRequest(new Request("GET", "_cat/nodes?h=http_address"));
        String body = new String(resp.getEntity().getContent().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        List<HttpHost> nodeHosts = new java.util.ArrayList<>();
        for (String line : body.split("\n")) {
            String hostPort = line.trim();
            if (hostPort.isEmpty()) continue;
            int colon = hostPort.lastIndexOf(':');
            String host = hostPort.substring(0, colon);
            int port = Integer.parseInt(hostPort.substring(colon + 1));
            nodeHosts.add(new HttpHost(getProtocol(), host, port));
        }
        log.info("resolved per-node HTTP hosts: {}", nodeHosts);
        return nodeHosts;
    }

    private IngestResult ingestOnNode(RestClient nodeClient, int dim, String index, CountDownLatch startGate) throws Exception {
        startGate.await();
        Request r = new Request("POST", "/" + index + "/_doc?refresh=true");
        r.setJsonEntity("{\"emb\": " + numericArray(dim) + "}");
        try {
            Response resp = nodeClient.performRequest(r);
            log.info("dim={} status={}", dim, resp.getStatusLine().getStatusCode());
            return new IngestResult(dim, resp.getStatusLine().getStatusCode(), "");
        } catch (ResponseException e) {
            String body = EntityUtils.toString(e.getResponse().getEntity());
            log.info("dim={} status={} body={}", dim, e.getResponse().getStatusLine().getStatusCode(), body);
            return new IngestResult(dim, e.getResponse().getStatusLine().getStatusCode(), body);
        }
    }

    private record IngestResult(int dim, int status, String errorBody) {
        boolean ok() {
            return status >= 200 && status < 300;
        }
    }
}

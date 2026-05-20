/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class LuceneSQCompressionIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    @SuppressWarnings("unchecked")
    public void testRollingUpgrade_luceneSQ() throws Exception {
        if (isLuceneSQSupported(getBWCVersion()) == false) {
            return;
        }
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .startObject("method")
                    .field(NAME, METHOD_HNSW)
                    .field(KNN_ENGINE, LUCENE_NAME)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;

            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> mixedMappings = getIndexMappingAsMap(testIndex);
                Map<String, Object> mixedProperties = (Map<String, Object>) mixedMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null", mixedProperties);
                Map<String, Object> mixedFieldProps = (Map<String, Object>) mixedProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null", mixedFieldProps);
                assertEquals(CompressionLevel.x32.getName(), mixedFieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                assertEquals(Mode.ON_DISK.getName(), mixedFieldProps.get(MODE_PARAMETER));
                break;

            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> upgradedMappings = getIndexMappingAsMap(testIndex);
                Map<String, Object> upgradedProperties = (Map<String, Object>) upgradedMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null after upgrade", upgradedProperties);
                Map<String, Object> upgradedFieldProps = (Map<String, Object>) upgradedProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null after upgrade", upgradedFieldProps);
                assertEquals(CompressionLevel.x32.getName(), upgradedFieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                assertEquals(Mode.ON_DISK.getName(), upgradedFieldProps.get(MODE_PARAMETER));

                deleteKNNIndex(testIndex);
        }
    }

    private boolean isLuceneSQSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(Version.V_3_6_0);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.apache.lucene.util.Version;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class LuceneTests extends KNNTestCase {

    public void testLucenHNSWMethod() throws IOException {
        KNNMethod luceneHNSW = Lucene.INSTANCE.getMethod(METHOD_HNSW);
        assertNotNull(luceneHNSW);

        int efConstruction = 100;
        int m = 17;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .field(METHOD_PARAMETER_M, m)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        assertNull(luceneHNSW.validate(knnMethodContext1));

        // Invalid parameter
        String invalidParameter = "invalid";
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field(invalidParameter, 10)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);
        assertNotNull(luceneHNSW.validate(knnMethodContext2));

        // Valid parameter, invalid value
        int invalidEfConstruction = -1;
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, invalidEfConstruction)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext3 = KNNMethodContext.parse(in);
        assertNotNull(luceneHNSW.validate(knnMethodContext3));

        // Unsupported space type
        SpaceType invalidSpaceType = SpaceType.LINF; // Not currently supported
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, invalidSpaceType.getValue())
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext4 = KNNMethodContext.parse(in);
        assertNotNull(luceneHNSW.validate(knnMethodContext4));

        // Check INNER_PRODUCT is supported with Lucene Engine
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .field(METHOD_PARAMETER_M, m)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext5 = KNNMethodContext.parse(in);
        assertNull(luceneHNSW.validate(knnMethodContext5));
    }

    public void testGetExtension() {
        Lucene luceneLibrary = new Lucene(Collections.emptyMap(), "", Collections.emptyMap());
        expectThrows(UnsupportedOperationException.class, luceneLibrary::getExtension);
    }

    public void testGetCompundExtension() {
        Lucene luceneLibrary = new Lucene(Collections.emptyMap(), "", Collections.emptyMap());
        expectThrows(UnsupportedOperationException.class, luceneLibrary::getCompoundExtension);
    }

    public void testScore() {
        Lucene luceneLibrary = new Lucene(Collections.emptyMap(), "", Collections.emptyMap());
        float rawScore = 10.0f;
        assertEquals(rawScore, luceneLibrary.score(rawScore, SpaceType.DEFAULT), 0.001);
    }

    public void testIsInitialized() {
        Lucene luceneLibrary = new Lucene(Collections.emptyMap(), "", Collections.emptyMap());
        assertFalse(luceneLibrary.isInitialized());
    }

    public void testSetInitialized() {
        Lucene luceneLibrary = new Lucene(Collections.emptyMap(), "", Collections.emptyMap());
        luceneLibrary.setInitialized(true);
        assertTrue(luceneLibrary.isInitialized());
    }

    public void testVersion() {
        Lucene luceneInstance = Lucene.INSTANCE;
        assertEquals(Version.LATEST.toString(), luceneInstance.getVersion());
    }

    public void testMmapFileExtensions() {
        final List<String> luceneMmapExtensions = Lucene.INSTANCE.mmapFileExtensions();
        assertNotNull(luceneMmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(luceneMmapExtensions));
        assertTrue(luceneMmapExtensions.containsAll(expectedSettings));
    }
}

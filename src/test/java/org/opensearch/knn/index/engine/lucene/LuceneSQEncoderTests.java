/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.CONFIDENCE_INTERVAL_DEPRECATION_MSG;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;

public class LuceneSQEncoderTests extends KNNTestCase {

    @Override
    protected boolean enableWarningsCheck() {
        return true;
    }

    public void testCalculateCompressionLevel() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(null, null));
    }

    public void testDeprecationWarning_whenConfidenceIntervalProvided() {
        Parameter<?> ciParam = new LuceneSQEncoder().getMethodComponent().getParameters().get(LUCENE_SQ_CONFIDENCE_INTERVAL);
        ciParam.validate(0.95, buildConfigContext());
        assertWarnings(String.format(CONFIDENCE_INTERVAL_DEPRECATION_MSG.replace("{}", "%s"), LUCENE_SQ_CONFIDENCE_INTERVAL, ENCODER_SQ));
    }

    public void testNoDeprecationWarning_whenConfidenceIntervalOmitted() {
        new LuceneSQEncoder().getMethodComponent()
            .validate(new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()), buildConfigContext());
        // assertWarnings is not called — OpenSearchTestCase will fail if any unexpected warnings are emitted
    }

    private KNNMethodConfigContext buildConfigContext() {
        return KNNMethodConfigContext.builder().dimension(128).versionCreated(Version.CURRENT).vectorDataType(VectorDataType.FLOAT).build();
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.ByteScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import java.io.IOException;

public class ByteScalarQuantizerTests extends KNNTestCase {
    public void testTrain() throws IOException {
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        String parameterString =
            "{\"index_description\":\"HNSW16,SQ8\",\"spaceType\":\"l2\",\"name\":\"hnsw\",\"data_type\":\"float\",\"parameters\":{\"ef_search\":256,\"ef_construction\":256,\"encoder\":{\"name\":\"sq\",\"parameters\":{\"clip\":false}}}}";
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test-field")
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNNConstants.PARAMETERS, parameterString)
            .build();

        ByteScalarQuantizer byteScalarQuantizer = new ByteScalarQuantizer(8);
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
        TrainingRequest<float[]> request = new ByteScalarQuantizerTests.MockTrainingRequest(params, vectors);
        QuantizationState state = byteScalarQuantizer.train(request, fieldInfo);

        assertTrue(state instanceof ByteScalarQuantizationState);
        ByteScalarQuantizationState byteScalarQuantizationState = (ByteScalarQuantizationState) state;
        assertNotNull(byteScalarQuantizationState.getIndexTemplate());
    }

    // Mock classes for testing
    private static class MockTrainingRequest extends TrainingRequest<float[]> {
        private final float[][] vectors;

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors) {
            super(vectors.length);
            this.vectors = vectors;
        }

        @Override
        public float[] getVectorAtThePosition(int position) {
            return vectors[position];
        }
    }
}

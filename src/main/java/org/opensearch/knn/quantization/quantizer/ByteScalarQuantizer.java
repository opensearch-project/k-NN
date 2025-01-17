/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.ByteScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;

public class ByteScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int bitsPerCoordinate;
    private final int samplingSize; // Sampling size for training
    private final Sampler sampler; // Sampler for training
    private static final int DEFAULT_SAMPLE_SIZE = 25000;

    public ByteScalarQuantizer(final int bitsPerCoordinate) {
        if (bitsPerCoordinate != 8) {
            throw new IllegalArgumentException("bitsPerCoordinate must be 8 for byte scalar quantizer.");
        }
        this.bitsPerCoordinate = bitsPerCoordinate;
        this.samplingSize = DEFAULT_SAMPLE_SIZE;
        this.sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
    }

    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) throws IOException {
        return null;
    }

    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest, final FieldInfo fieldInfo) throws IOException {
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        if (sampledIndices.length == 0) {
            return null;
        }
        float[] vector = trainingRequest.getVectorAtThePosition(sampledIndices[0]);
        if (vector == null) {
            throw new IllegalArgumentException("Vector at sampled index " + sampledIndices[0] + " is null.");
        }
        int dimension = vector.length;

        try (
            final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(
                extractVectorDataType(fieldInfo),
                4 * dimension,
                sampledIndices.length
            )
        ) {
            for (int i = 0; i < sampledIndices.length; i++) {
                Object vectorToTransfer = trainingRequest.getVectorAtThePosition(sampledIndices[i]);
                vectorTransfer.transfer(vectorToTransfer, true);
            }
            vectorTransfer.flush(true);

            byte[] indexTemplate = JNIService.trainIndex(
                getParameters(fieldInfo),
                dimension,
                vectorTransfer.getVectorAddress(),
                KNNEngine.FAISS
            );
            ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
            return new ByteScalarQuantizationState(params, indexTemplate);
        }
    }

    private Map<String, Object> getParameters(final FieldInfo fieldInfo) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));

            Map<String, Object> algoParams = new HashMap<>();

            String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
            if (efConstruction != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
            }

            String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
            if (m != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
            }
            parameters.put(PARAMETERS, algoParams);
        } else {
            parameters.putAll(
                XContentHelper.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new BytesArray(parametersString),
                    MediaTypeRegistry.getDefaultMediaType()
                ).map()
            );
        }

        parameters.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT);

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));

        return parameters;
    }

    @Override
    public void quantize(float[] vector, QuantizationState state, QuantizationOutput<byte[]> output) {
        // Quantization is performed inside Faiss Scalar Quantizer
    }
}

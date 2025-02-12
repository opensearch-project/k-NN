/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.training;

import org.apache.commons.lang.ArrayUtils;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.quantization.factory.QuantizerFactory;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.Quantizer;
import org.opensearch.search.SearchHit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Transfers float vectors from JVM to native memory.
 */
public class FloatTrainingDataConsumer extends TrainingDataConsumer {

    private final QuantizationConfig quantizationConfig;

    /**
     * Constructor
     *
     * @param trainingDataAllocation NativeMemoryAllocation that contains information about native memory allocation.
     */
    public FloatTrainingDataConsumer(NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation) {
        super(trainingDataAllocation);
        this.quantizationConfig = trainingDataAllocation.getQuantizationConfig();
    }

    @Override
    public void accept(List<?> floats) {
        if (isValidFloatsAndQuantizationConfig(floats)) {
            try {
                List<byte[]> byteVectors = quantizeVectors(floats);
                long memoryAddress = trainingDataAllocation.getMemoryAddress();
                memoryAddress = JNICommons.storeBinaryVectorData(memoryAddress, byteVectors.toArray(new byte[0][0]), byteVectors.size());
                trainingDataAllocation.setMemoryAddress(memoryAddress);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else {
            trainingDataAllocation.setMemoryAddress(
                JNICommons.storeVectorData(
                    trainingDataAllocation.getMemoryAddress(),
                    floats.stream().map(v -> ArrayUtils.toPrimitive((Float[]) v)).toArray(float[][]::new),
                    floats.size()
                )
            );
        }
    }

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<Float[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);
            if (!(fieldValue instanceof List<?>)) {
                continue;
            }

            List<Number> fieldList = (List<Number>) fieldValue;
            vectors.add(fieldList.stream().map(Number::floatValue).toArray(Float[]::new));
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }

    private List<byte[]> quantizeVectors(List<?> vectors) throws IOException {
        List<byte[]> bytes = new ArrayList<>();
        ScalarQuantizationParams quantizationParams = new ScalarQuantizationParams(quantizationConfig.getQuantizationType());
        Quantizer<float[], byte[]> quantizer = QuantizerFactory.getQuantizer(quantizationParams);
        // Create training request
        TrainingRequest<float[]> trainingRequest = new TrainingRequest<float[]>(vectors.size()) {
            @Override
            public float[] getVectorAtThePosition(int position) {
                return ArrayUtils.toPrimitive((Float[]) vectors.get(position));
            }

            @Override
            public void resetVectorValues() {
                // No-op
            }

        };
        QuantizationState quantizationState = quantizer.train(trainingRequest);
        BinaryQuantizationOutput binaryQuantizationOutput = new BinaryQuantizationOutput(quantizationConfig.getQuantizationType().getId());
        for (int i = 0; i < vectors.size(); i++) {
            quantizer.quantize(ArrayUtils.toPrimitive((Float[]) vectors.get(i)), quantizationState, binaryQuantizationOutput);
            bytes.add(binaryQuantizationOutput.getQuantizedVectorCopy());
        }

        return bytes;
    }

    private boolean isValidFloatsAndQuantizationConfig(List<?> floats) {
        return floats != null && floats.isEmpty() == false && quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY;
    }
}

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

package org.opensearch.knn.index;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesConsumer;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.quantization.QuantizationManager;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.quantizer.Quantizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This is a single layer that will be responsible for creating the native indices. Right now this is just a POC code,
 * this needs to be fixed. Its more of a testing to see if everything works correctly.
 */
@Log4j2
public class NativeIndexCreationManager {

    public static void startIndexCreation(
        final SegmentWriteState segmentWriteState,
        final KNNVectorValues<float[]> vectorValues,
        final FieldInfo fieldInfo
    ) throws IOException {
        KNNCodecUtil.Pair pair = streamFloatVectors(vectorValues);
        if (pair.getVectorAddress() == 0 || pair.docs.length == 0) {
            log.info("Skipping engine index creation as there are no vectors or docs in the segment");
            return;
        }
        createNativeIndex(segmentWriteState, fieldInfo, pair);
    }

    private static void createNativeIndex(
        final SegmentWriteState segmentWriteState,
        final FieldInfo fieldInfo,
        final KNNCodecUtil.Pair pair
    ) throws IOException {
        KNN80DocValuesConsumer.createNativeIndex(segmentWriteState, fieldInfo, pair);
    }

    private static KNNCodecUtil.Pair streamFloatVectors(final KNNVectorValues<float[]> kNNVectorValues) throws IOException {
        List<byte[]> vectorList = new ArrayList<>();
        List<Integer> docIdList = new ArrayList<>();
        long vectorAddress = 0;
        int dimension = 0;
        long totalLiveDocs = kNNVectorValues.totalLiveDocs();
        long vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        long vectorsPerTransfer = Integer.MIN_VALUE;

        QuantizationParams params = getQuantizationParams(); // Implement this method to get appropriate params
        Quantizer<float[], byte[]> quantizer = (Quantizer<float[], byte[]>) QuantizationManager.getInstance().getQuantizer(params);

        KNNVectorValuesIterator iterator = kNNVectorValues.getVectorValuesIterator();

        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            byte[] quantizedVector = quantizer.quantize(kNNVectorValues.getVector()).getQuantizedVector();
            dimension = kNNVectorValues.getVector().length;
            if (vectorsPerTransfer == Integer.MIN_VALUE) {
                vectorsPerTransfer = (dimension * Byte.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
                // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
                // Doing this will reduce 1 extra trip to JNI layer.
                if (vectorsPerTransfer == 0) {
                    vectorsPerTransfer = totalLiveDocs;
                }
            }

            if (vectorList.size() == vectorsPerTransfer) {
                vectorAddress = JNICommons.storeByteVectorData(
                    vectorAddress,
                    vectorList.toArray(new byte[][] {}),
                    totalLiveDocs * dimension
                );
                // We should probably come up with a better way to reuse the vectorList memory which we have
                // created. Problem here is doing like this can lead to a lot of list memory which is of no use and
                // will be garbage collected later on, but it creates pressure on JVM. We should revisit this.
                vectorList = new ArrayList<>();
            }

            vectorList.add(quantizedVector);
            docIdList.add(doc);
        }

        if (vectorList.isEmpty() == false) {
            vectorAddress = JNICommons.storeByteVectorData(vectorAddress, vectorList.toArray(new byte[][] {}), totalLiveDocs * dimension);
        }
        // SerializationMode.COLLECTION_OF_FLOATS is not getting used. I just added it to ensure code successfully
        // works.
        return new KNNCodecUtil.Pair(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorAddress,
            dimension,
            SerializationMode.COLLECTION_OF_FLOATS
        );
    }

    private static QuantizationParams getQuantizationParams() {
        // Implement this method to return appropriate quantization parameters based on your use case
        return new SQParams(SQTypes.ONE_BIT); // Example, modify as needed
    }
}

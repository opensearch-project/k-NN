/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.quantizationService.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNVectorUtil.intListToArray;
import static org.opensearch.knn.common.KNNVectorUtil.iterateVectorValuesOnce;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;

/**
 * Transfers all vectors to off heap and then builds an index
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class DefaultIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static DefaultIndexBuildStrategy INSTANCE = new DefaultIndexBuildStrategy();

    public static DefaultIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    /**
     * Builds and writes a k-NN index using the provided vector values and index parameters. This method handles both
     * quantized and non-quantized vectors, transferring them off-heap before building the index using native JNI services.
     *
     * <p>The method first iterates over the vector values to calculate the necessary bytes per vector. If quantization is
     * enabled, the vectors are quantized before being transferred off-heap. Once all vectors are transferred, they are
     * flushed and used to build the index. The index is then written to the specified path using JNI calls.</p>
     *
     * @param indexInfo        The {@link BuildIndexParams} containing the parameters and configuration for building the index.
     * @param knnVectorValues  The {@link KNNVectorValues} representing the vectors to be indexed.
     * @throws IOException     If an I/O error occurs during the process of building and writing the index.
     */
    public void buildAndWriteIndex(final BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException {
        // Needed to make sure we don't get 0 dimensions while initializing index
        iterateVectorValuesOnce(knnVectorValues);
        QuantizationService quantizationHandler = QuantizationService.getInstance();
        QuantizationState quantizationState = indexInfo.getQuantizationState();
        QuantizationOutput quantizationOutput = null;

        int bytesPerVector;
        int dimensions;

        // Handle quantization state if present
        if (quantizationState != null) {
            bytesPerVector = quantizationState.getBytesPerVector();
            dimensions = quantizationState.getDimensions();
            quantizationOutput = quantizationHandler.createQuantizationOutput(quantizationState.getQuantizationParams());
        } else {
            bytesPerVector = knnVectorValues.bytesPerVector();
            dimensions = knnVectorValues.dimension();
        }

        int transferLimit = (int) Math.max(1, KNNSettings.getVectorStreamingMemoryLimit().getBytes() / bytesPerVector);
        try (final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType(), transferLimit)) {
            final List<Integer> transferredDocIds = new ArrayList<>((int) knnVectorValues.totalLiveDocs());

            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                if (quantizationState != null && quantizationOutput != null) {
                    quantizationHandler.quantize(quantizationState, knnVectorValues.getVector(), quantizationOutput);
                    vectorTransfer.transfer(quantizationOutput.getQuantizedVector(), true);
                } else {
                    vectorTransfer.transfer(knnVectorValues.conditionalCloneVector(), true);
                }
                // append is true here so off heap memory buffer isn't overwritten
                transferredDocIds.add(knnVectorValues.docId());
                knnVectorValues.nextDoc();
            }
            vectorTransfer.flush(true);

            final Map<String, Object> params = indexInfo.getParameters();
            long vectorAddress = vectorTransfer.getVectorAddress();
            // Currently this is if else as there are only two cases, with more cases this will have to be made
            // more maintainable
            if (params.containsKey(MODEL_ID)) {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndexFromTemplate(
                        intListToArray(transferredDocIds),
                        vectorAddress,
                        dimensions,
                        indexInfo.getIndexPath(),
                        (byte[]) params.get(KNNConstants.MODEL_BLOB_PARAMETER),
                        params,
                        indexInfo.getKnnEngine()
                    );
                    return null;
                });
            } else {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.createIndex(
                        intListToArray(transferredDocIds),
                        vectorAddress,
                        dimensions,
                        indexInfo.getIndexPath(),
                        params,
                        indexInfo.getKnnEngine()
                    );
                    return null;
                });
            }
            // Resetting here as vectors are deleted in JNILayer for non-iterative index builds
            vectorTransfer.reset();
        } catch (Exception exception) {
            throw new RuntimeException(
                "Failed to build index, field name " + indexInfo.getFieldName() + ", parameters " + indexInfo,
                exception
            );
        }
    }
}

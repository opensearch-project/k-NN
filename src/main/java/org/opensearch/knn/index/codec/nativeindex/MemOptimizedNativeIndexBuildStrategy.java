/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.models.quantizationState.ByteScalarQuantizationState;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNVectorUtil.intListToArray;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Iteratively builds the index. Iterative builds are memory optimized as it does not require all vectors
 * to be transferred. It transfers vectors in small batches, builds index and can clear the offheap space where
 * the vectors were transferred
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class MemOptimizedNativeIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static MemOptimizedNativeIndexBuildStrategy INSTANCE = new MemOptimizedNativeIndexBuildStrategy();

    public static MemOptimizedNativeIndexBuildStrategy getInstance() {
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
     * @throws IOException     If an I/O error occurs during the process of building and writing the index.
     */
    public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException {
        final KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        // Needed to make sure we don't get 0 dimensions while initializing index
        initializeVectorValues(knnVectorValues);
        KNNEngine engine = indexInfo.getKnnEngine();
        Map<String, Object> indexParameters = indexInfo.getParameters();
        IndexBuildSetup indexBuildSetup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, indexInfo);
        long indexMemoryAddress;

        if (isTemplate(indexInfo)) {
            // Initialize the index from Template
            indexMemoryAddress = AccessController.doPrivileged(
                (PrivilegedAction<Long>) () -> JNIService.initIndexFromTemplate(
                    indexInfo.getTotalLiveDocs(),
                    indexBuildSetup.getDimensions(),
                    indexParameters,
                    engine,
                    getIndexTemplate(indexInfo)
                )
            );

        } else {
            // Initialize the index
            indexMemoryAddress = AccessController.doPrivileged(
                (PrivilegedAction<Long>) () -> JNIService.initIndex(
                    indexInfo.getTotalLiveDocs(),
                    indexBuildSetup.getDimensions(),
                    indexParameters,
                    engine
                )
            );
        }

        try (
            final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(
                indexInfo.getVectorDataType(),
                indexBuildSetup.getBytesPerVector(),
                indexInfo.getTotalLiveDocs()
            )
        ) {

            final List<Integer> transferredDocIds = new ArrayList<>(vectorTransfer.getTransferLimit());

            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                Object vector = QuantizationIndexUtils.processAndReturnVector(knnVectorValues, indexBuildSetup);
                // append is false to be able to reuse the memory location
                boolean transferred = vectorTransfer.transfer(vector, false);
                transferredDocIds.add(knnVectorValues.docId());
                if (transferred) {
                    // Insert vectors
                    long vectorAddress = vectorTransfer.getVectorAddress();
                    AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                        JNIService.insertToIndex(
                            intListToArray(transferredDocIds),
                            vectorAddress,
                            indexBuildSetup.getDimensions(),
                            indexParameters,
                            indexMemoryAddress,
                            engine
                        );
                        return null;
                    });
                    transferredDocIds.clear();
                }
                knnVectorValues.nextDoc();
            }

            boolean flush = vectorTransfer.flush(false);
            // Need to make sure that the flushed vectors are indexed
            if (flush) {
                long vectorAddress = vectorTransfer.getVectorAddress();
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.insertToIndex(
                        intListToArray(transferredDocIds),
                        vectorAddress,
                        indexBuildSetup.getDimensions(),
                        indexParameters,
                        indexMemoryAddress,
                        engine
                    );
                    return null;
                });
                transferredDocIds.clear();
            }

            // Write vector
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.writeIndex(indexInfo.getIndexOutputWithBuffer(), indexMemoryAddress, engine, indexParameters);
                return null;
            });

        } catch (Exception exception) {
            throw new RuntimeException(
                "Failed to build index, field name [" + indexInfo.getFieldName() + "], parameters " + indexInfo,
                exception
            );
        }
    }

    private static boolean isTemplate(final BuildIndexParams indexInfo) {
        return (indexInfo.getQuantizationState() instanceof ByteScalarQuantizationState);
    }

    private byte[] getIndexTemplate(BuildIndexParams indexInfo) {
        ByteScalarQuantizationState byteSQState = (ByteScalarQuantizationState) indexInfo.getQuantizationState();
        return byteSQState.getIndexTemplate();
    }
}

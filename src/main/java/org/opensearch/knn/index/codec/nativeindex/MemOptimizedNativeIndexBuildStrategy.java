/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapByteQuantizedVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

/**
 * Iteratively builds the index.
 */
final class MemOptimizedNativeIndexBuildStrategy implements NativeIndexBuildStrategy {

    private static MemOptimizedNativeIndexBuildStrategy INSTANCE = new MemOptimizedNativeIndexBuildStrategy();

    public static MemOptimizedNativeIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    private MemOptimizedNativeIndexBuildStrategy() {}

    public void buildAndWriteIndex(BuildIndexParams indexInfo, final KNNVectorValues<?> knnVectorValues) throws IOException {
        // Needed to make sure we dont get 0 dimensions while initializing index
        knnVectorValues.init();
        KNNEngine engine = indexInfo.getKnnEngine();
        Map<String, Object> indexParameters = indexInfo.getParameters();

        // Initialize the index
        long indexMemoryAddress = AccessController.doPrivileged(
            (PrivilegedAction<Long>) () -> JNIService.initIndexFromScratch(
                knnVectorValues.totalLiveDocs(),
                knnVectorValues.dimension(),
                indexParameters,
                engine
            )
        );

        try (final VectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType(), knnVectorValues)) {

            while (vectorTransfer.hasNext()) {
                vectorTransfer.transferBatch();
                long vectorAddress = vectorTransfer.getVectorAddress();
                int[] docs = vectorTransfer.getTransferredDocsIds();

                // Insert vectors
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.insertToIndex(docs, vectorAddress, knnVectorValues.dimension(), indexParameters, indexMemoryAddress, engine);
                    return null;
                });
            }

            // Write vector
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.writeIndex(indexInfo.getIndexPath(), indexMemoryAddress, engine, indexParameters);
                return null;
            });

        } catch (Exception exception) {
            throw new RuntimeException("Failed to build index", exception);
        }
    }

    // TODO: Will probably need a factory once quantization is added
    private VectorTransfer getVectorTransfer(VectorDataType vectorDataType, KNNVectorValues<?> knnVectorValues) throws IOException {
        switch (vectorDataType) {
            case FLOAT:
                return new OffHeapFloatVectorTransfer((KNNFloatVectorValues) knnVectorValues);
            case BINARY:
            case BYTE:
                return new OffHeapByteQuantizedVectorTransfer<>((KNNVectorValues<byte[]>) knnVectorValues);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}

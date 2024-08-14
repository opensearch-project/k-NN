/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapByteVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

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

        //TODO: compute transferLimit here based on KNNSettings and Quantization parameters
        try (final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(indexInfo.getVectorDataType())) {

            List<Integer> tranferredDocIds = new ArrayList<>();
            while (knnVectorValues.docId() != NO_MORE_DOCS) {
                //TODO: Quantization fits here
                boolean transferred = vectorTransfer.transfer(knnVectorValues.getVector());
                tranferredDocIds.add(knnVectorValues.docId());
                if (transferred) {
                    // Insert vectors
                    long vectorAddress = vectorTransfer.getVectorAddress();
                    AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                        JNIService.insertToIndex(tranferredDocIds.stream().mapToInt(i -> i).toArray(), vectorAddress, knnVectorValues.dimension(), indexParameters, indexMemoryAddress, engine);
                        return null;
                    });
                    tranferredDocIds.clear();
                }
                knnVectorValues.nextDoc();
            }

            boolean flush = vectorTransfer.flush();
            //Need to make sure that the flushed vectors are indexed
            if (flush) {
                long vectorAddress = vectorTransfer.getVectorAddress();
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    JNIService.insertToIndex(tranferredDocIds.stream().mapToInt(i -> i).toArray(), vectorAddress, knnVectorValues.dimension(), indexParameters, indexMemoryAddress, engine);
                    return null;
                });
                tranferredDocIds.clear();
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
    private <T> OffHeapVectorTransfer<T> getVectorTransfer(VectorDataType vectorDataType) throws IOException {
        switch (vectorDataType) {
            case FLOAT:
                return (OffHeapVectorTransfer<T>) new OffHeapFloatVectorTransfer();
            case BINARY:
            case BYTE:
                return (OffHeapVectorTransfer<T>) new OffHeapByteVectorTransfer(100);
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}

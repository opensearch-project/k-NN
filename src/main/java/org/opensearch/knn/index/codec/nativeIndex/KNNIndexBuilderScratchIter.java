/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.builder;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import org.apache.lucene.index.BinaryDocValues;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;

public class KNNIndexBuilderScratchIter extends KNNIndexBuilderScratch {

    @Override
    protected void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException {
        long indexAddress = initIndexFromScratch(indexInfo.numDocs, indexInfo.vectorInfo.dimension, indexInfo.knnEngine, indexInfo.parameters);
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(indexInfo.vectorInfo.vectorDataType), true);
        try {
            for (; !batch.finished; batch = KNNCodecUtil.getVectorBatch(values, getVectorTransfer(indexInfo.vectorInfo.vectorDataType), true)) {
                insertToIndex(batch, indexInfo.knnEngine, indexAddress, indexInfo.parameters);
            }
            insertToIndex(batch, indexInfo.knnEngine, indexAddress, indexInfo.parameters);
            writeIndex(indexAddress, indexInfo.indexPath, indexInfo.knnEngine, indexInfo.parameters);
        } catch (Exception e) {
            JNIService.free(indexAddress, indexInfo.knnEngine, VectorDataType.BINARY == indexInfo.vectorInfo.vectorDataType);
            if (batch.getVectorAddress() != 0) {
                JNICommons.freeVectorData(batch.getVectorAddress());
            }
            throw e;
        }
    }

    private long initIndexFromScratch(long size, int dim, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        return AccessController.doPrivileged((PrivilegedAction<Long>) () -> {
            return JNIService.initIndexFromScratch(size, dim, parameters, knnEngine);
        });
    }

    private void insertToIndex(KNNCodecUtil.VectorBatch batch, KNNEngine knnEngine, long indexAddress, Map<String, Object> parameters)
        throws IOException {
        if (batch.docs.length == 0) {
            logger.debug("Index insertion called with a batch without docs.");
            return;
        }
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.insertToIndex(batch.docs, batch.getVectorAddress(), batch.getDimension(), parameters, indexAddress, knnEngine);
            return null;
        });
    }

    private void writeIndex(long indexAddress, String indexPath, KNNEngine knnEngine, Map<String, Object> parameters) throws IOException {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.writeIndex(indexPath, indexAddress, knnEngine, parameters);
            return null;
        });
    }
}

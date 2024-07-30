/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

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

/**
 * Class to build the KNN index from scratch iteratively and write it to disk
 */
public class NativeIndexWriterScratchIter extends NativeIndexWriterScratch {

    @Override
    protected void createIndex(NativeIndexInfo indexInfo, BinaryDocValues values) throws IOException {
        long indexAddress = initIndexFromScratch(
            indexInfo.getNumDocs(),
            indexInfo.getVectorInfo().getDimension(),
            indexInfo.getKnnEngine(),
            indexInfo.getParameters()
        );
        KNNCodecUtil.VectorBatch batch = KNNCodecUtil.getVectorBatch(
            values,
            getVectorTransfer(indexInfo.getVectorInfo().getVectorDataType()),
            true
        );
        try {
            for (; batch.finished == false; batch = KNNCodecUtil.getVectorBatch(
                values,
                getVectorTransfer(indexInfo.getVectorInfo().getVectorDataType()),
                true
            )) {
                insertToIndex(batch, indexInfo.getKnnEngine(), indexAddress, indexInfo.getParameters());
            }
            insertToIndex(batch, indexInfo.getKnnEngine(), indexAddress, indexInfo.getParameters());
            writeIndex(indexAddress, indexInfo.getIndexPath(), indexInfo.getKnnEngine(), indexInfo.getParameters());
        } catch (Exception e) {
            JNIService.free(indexAddress, indexInfo.getKnnEngine(), VectorDataType.BINARY == indexInfo.getVectorInfo().getVectorDataType());
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

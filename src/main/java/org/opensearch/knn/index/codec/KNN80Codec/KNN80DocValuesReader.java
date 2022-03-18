/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.EmptyDocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;

import java.util.ArrayList;
import java.util.List;

/**
 * Reader for KNNDocValues from the segments
 */
class KNN80DocValuesReader extends EmptyDocValuesProducer {

    private final MergeState mergeState;

    KNN80DocValuesReader(MergeState mergeState) {
        this.mergeState = mergeState;
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) {
        try {
            List<BinaryDocValuesSub> subs = new ArrayList<>(this.mergeState.docValuesProducers.length);
            for (int i = 0; i < this.mergeState.docValuesProducers.length; i++) {
                BinaryDocValues values = null;
                DocValuesProducer docValuesProducer = mergeState.docValuesProducers[i];
                if (docValuesProducer != null) {
                    FieldInfo readerFieldInfo = mergeState.fieldInfos[i].fieldInfo(field.name);
                    if (readerFieldInfo != null && readerFieldInfo.getDocValuesType() == DocValuesType.BINARY) {
                        values = docValuesProducer.getBinary(readerFieldInfo);
                    }
                    if (values != null) {
                        subs.add(new BinaryDocValuesSub(mergeState.docMaps[i], values));
                    }
                }
            }
            return new KNN80BinaryDocValues(DocIDMerger.of(subs, mergeState.needsIndexSort));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}

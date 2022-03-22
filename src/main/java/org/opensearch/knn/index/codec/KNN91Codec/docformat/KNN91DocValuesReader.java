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

package org.opensearch.knn.index.codec.KNN91Codec.docformat;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.EmptyDocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;

import java.util.ArrayList;
import java.util.List;

/**
 * Reader for KNNDocValues from the segments
 */
class KNN91DocValuesReader extends EmptyDocValuesProducer {

    private final MergeState mergeState;

    KNN91DocValuesReader(MergeState mergeState) {
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
            return new KNN91BinaryDocValues(DocIDMerger.of(subs, mergeState.needsIndexSort));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}

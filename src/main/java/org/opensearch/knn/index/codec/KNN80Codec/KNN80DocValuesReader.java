/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.EmptyDocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for KNNDocValues from the segments
 */
@Log4j2
class KNN80DocValuesReader extends EmptyDocValuesProducer {

    private final MergeState mergeState;

    KNN80DocValuesReader(MergeState mergeState) {
        this.mergeState = mergeState;
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) {
        long totalLiveDocs = 0;
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
                        totalLiveDocs = totalLiveDocs + getLiveDocsCount(values, this.mergeState.liveDocs[i]);
                        // docValues will be consumed when liveDocs are not null, hence resetting the docsValues
                        // pointer.
                        values = this.mergeState.liveDocs[i] != null ? docValuesProducer.getBinary(readerFieldInfo) : values;

                        subs.add(new BinaryDocValuesSub(mergeState.docMaps[i], values));
                    }
                }
            }
            return new KNN80BinaryDocValues(DocIDMerger.of(subs, mergeState.needsIndexSort)).setTotalLiveDocs(totalLiveDocs);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This function return the liveDocs count present in the BinaryDocValues. If the liveDocsBits is null, then we
     * can use {@link BinaryDocValues#cost()} function to get max docIds. But if LiveDocsBits is not null, then we
     * iterate over the BinaryDocValues and validate if the docId is present in the live docs bits or not.
     *
     * @param binaryDocValues {@link BinaryDocValues}
     * @param liveDocsBits {@link Bits}
     * @return total number of liveDocs.
     * @throws IOException
     */
    private long getLiveDocsCount(final BinaryDocValues binaryDocValues, final Bits liveDocsBits) throws IOException {
        long liveDocs = 0;
        if (liveDocsBits != null) {
            int docId;
            // This is not the right way to log the time. I create a github issue for adding an annotation to track
            // the time. https://github.com/opensearch-project/k-NN/issues/1594
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            for (docId = binaryDocValues.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = binaryDocValues.nextDoc()) {
                if (liveDocsBits.get(docId)) {
                    liveDocs++;
                }
            }
            stopWatch.stop();
            log.debug("Time taken to iterate over binary doc values: {} ms", stopWatch.totalTime().millis());
        } else {
            liveDocs = binaryDocValues.cost();
        }
        return liveDocs;
    }
}

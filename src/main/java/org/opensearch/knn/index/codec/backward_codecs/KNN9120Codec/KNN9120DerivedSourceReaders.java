/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.Nullable;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Class holds the readers necessary to implement derived source. Important to note that if a segment does not have
 * any of these fields, the values will be null. Caller needs to check if these are null before using.
 */
@RequiredArgsConstructor
@Getter
public class KNN9120DerivedSourceReaders implements Closeable {
    @Nullable
    private final KnnVectorsReader knnVectorsReader;
    @Nullable
    private final DocValuesProducer docValuesProducer;
    @Nullable
    private final FieldsProducer fieldsProducer;
    @Nullable
    private final NormsProducer normsProducer;

    // Copied from lucene (https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/index/SegmentCoreReaders.java):
    // We need to reference count these readers because they may be shared amongst different instances.
    // "Counts how many other readers share the core objects
    // (freqStream, proxStream, tis, etc.) of this reader;
    // when coreRef drops to 0, these core objects may be
    // closed. A given instance of SegmentReader may be
    // closed, even though it shares core objects with other
    // SegmentReaders":
    private final AtomicInteger ref = new AtomicInteger(1);

    /**
     * This method is used to clone the KNN9120DerivedSourceReaders object.
     * This is used when the object is passed to multiple threads.
     *
     * @return KNN9120DerivedSourceReaders object
     */
    public KNN9120DerivedSourceReaders cloneWithMerge() {
        // For cloning, we dont need to reference count. In Lucene, the merging will actually not close any of the
        // readers, so it should only be handled by the original code. See
        // https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L3372
        // for more details
        return this;
    }

    @Override
    public void close() throws IOException {
        decRef();
    }

    private void incRef() {
        int count;
        while ((count = ref.get()) > 0) {
            if (ref.compareAndSet(count, count + 1)) {
                return;
            }
        }
        throw new AlreadyClosedException("DerivedSourceReaders is already closed");
    }

    private void decRef() throws IOException {
        if (ref.decrementAndGet() == 0) {
            IOUtils.close(knnVectorsReader, docValuesProducer, fieldsProducer, normsProducer);
        }
    }
}

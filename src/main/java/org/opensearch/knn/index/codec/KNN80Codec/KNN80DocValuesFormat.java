/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.backward_codecs.lucene80.Lucene80DocValuesFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * Encodes/Decodes per document values
 */
public class KNN80DocValuesFormat extends DocValuesFormat {
    private final DocValuesFormat delegate;

    public KNN80DocValuesFormat() {
        this(new Lucene80DocValuesFormat());
    }

    /**
     * Constructor that takes delegate in order to handle non-overridden methods
     *
     * @param delegate DocValuesFormat to handle non-overridden methods
     */
    public KNN80DocValuesFormat(DocValuesFormat delegate) {
        super(delegate.getName());
        this.delegate = delegate;
    }

    @Override
    public DocValuesConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
        return new KNN80DocValuesConsumer(delegate.fieldsConsumer(state), state);
    }

    @Override
    public DocValuesProducer fieldsProducer(SegmentReadState state) throws IOException {
        return new KNN80DocValuesProducer(delegate.fieldsProducer(state), state);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;

/**
 * Factory abstraction for KNN format facade creation
 */
public class KNNFormatFactory {

    /**
     * Return facade class that abstracts format specific to KNN910 codec
     * @param delegate delegate codec that is wrapped by KNN codec
     * @return
     */
    public static KNNFormatFacade createKNN910Format(final Codec delegate) {
        final KNNFormatFacade knnFormatFacade = new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        );
        return knnFormatFacade;
    }

    /**
     * Return facade class that abstracts format specific to KNN920 codec
     * @param delegate delegate codec that is wrapped by KNN codec
     * @return
     */
    public static KNNFormatFacade createKNN920Format(final Codec delegate) {
        final KNNFormatFacade knnFormatFacade = new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        );
        return knnFormatFacade;
    }

    /**
     * Return facade class that abstracts format specific to KNN940 codec
     * @param delegate delegate codec that is wrapped by KNN codec
     */
    public static KNNFormatFacade createKNN940Format(final Codec delegate) {
        final KNNFormatFacade knnFormatFacade = new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        );
        return knnFormatFacade;
    }
}

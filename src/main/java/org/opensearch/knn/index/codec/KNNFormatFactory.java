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

    public static KNNFormatFacade createKNN910Format(final Codec delegate) {
        final KNNFormatFacade knnFormatFacade = new KNNFormatFacade(
            new KNN80DocValuesFormat(delegate.docValuesFormat()),
            new KNN80CompoundFormat(delegate.compoundFormat())
        );
        return knnFormatFacade;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN91Codec.docformat.KNN91CompoundFormat;
import org.opensearch.knn.index.codec.KNN91Codec.docformat.KNN91DocValuesFormat;

/**
 * Factory abstraction for KNN document format facades
 */
public class KNNFormatFactory {

    public static KNNFormatFacade createKNN91Format(final Codec delegate) {
        final KNNFormatFacade knnFormatFacade = new KNNFormatFacade(
            new KNN91DocValuesFormat(delegate.docValuesFormat()),
            new KNN91CompoundFormat(delegate.compoundFormat())
        );
        return knnFormatFacade;
    }
}

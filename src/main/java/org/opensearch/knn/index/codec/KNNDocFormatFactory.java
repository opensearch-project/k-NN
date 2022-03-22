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

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN91Codec.docformat.KNN91CompoundFormat;
import org.opensearch.knn.index.codec.KNN91Codec.KNN91DocFormat;
import org.opensearch.knn.index.codec.KNN91Codec.docformat.KNN91DocValuesFormat;

/**
 * Factory abstraction for KNN document format facades
 */
public class KNNDocFormatFactory {

    public static KNNDocFormatFacade createKNN91DocFormat(Codec delegate) {
        final KNNDocFormatFacade knnDocFormatFacade = new KNN91DocFormat(
            new KNN91DocValuesFormat(delegate.docValuesFormat()),
            new KNN91CompoundFormat(delegate.compoundFormat())
        );
        return knnDocFormatFacade;
    }
}

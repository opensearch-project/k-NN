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

package org.opensearch.knn.index.codec.KNN91Codec;

import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.opensearch.knn.index.codec.KNNDocFormatFacade;

public class KNN91DocFormat implements KNNDocFormatFacade {

    private final DocValuesFormat docValuesFormat;
    private final CompoundFormat compoundFormat;

    public KNN91DocFormat(final DocValuesFormat docValuesFormat, final CompoundFormat compoundFormat) {
        this.docValuesFormat = docValuesFormat;
        this.compoundFormat = compoundFormat;
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return docValuesFormat;
    }

    @Override
    public CompoundFormat compoundFormat() {
        return compoundFormat;
    }
}

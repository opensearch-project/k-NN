/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;

/**
 * Class abstracts facade for plugin formats.
 */
public class KNNFormatFacade {

    private final DocValuesFormat docValuesFormat;
    private final CompoundFormat compoundFormat;

    public KNNFormatFacade(final DocValuesFormat docValuesFormat, final CompoundFormat compoundFormat) {
        this.docValuesFormat = docValuesFormat;
        this.compoundFormat = compoundFormat;
    }

    public DocValuesFormat docValuesFormat() {
        return docValuesFormat;
    }

    public CompoundFormat compoundFormat() {
        return compoundFormat;
    }
}

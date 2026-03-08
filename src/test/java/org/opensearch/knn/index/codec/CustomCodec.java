/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;

public class CustomCodec extends FilterCodec {
    private final StoredFieldsFormat storedFieldsFormat;

    public CustomCodec() {
        super("CustomCodec", new Lucene104Codec());
        this.storedFieldsFormat = new CustomStoredFieldsFormat();
    }

    @Override
    public StoredFieldsFormat storedFieldsFormat() {
        return storedFieldsFormat;
    }
}

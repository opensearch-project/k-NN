/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceStoredFieldVisitor;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceVectorInjector;

import java.io.IOException;

@RequiredArgsConstructor
public class DerivedSourceStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader delegate;
    // Given docId and source, process source
    private final DerivedSourceVectorInjector derivedSourceVectorInjector;

    @Setter
    private boolean shouldInject = true;

    @Override
    public void document(int docId, StoredFieldVisitor storedFieldVisitor) throws IOException {
        if (shouldInject) {
            delegate.document(docId, new DerivedSourceStoredFieldVisitor(storedFieldVisitor, docId, derivedSourceVectorInjector));
            return;
        }
        delegate.document(docId, storedFieldVisitor);
    }

    @Override
    public StoredFieldsReader clone() {
        return new DerivedSourceStoredFieldsReader(delegate.clone(), derivedSourceVectorInjector);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    public static StoredFieldsReader wrapForMerge(StoredFieldsReader storedFieldsReader) {
        if (storedFieldsReader instanceof DerivedSourceStoredFieldsReader) {
            StoredFieldsReader storedFieldsReaderClone = storedFieldsReader.clone();
            ((DerivedSourceStoredFieldsReader) storedFieldsReaderClone).setShouldInject(false);
            return storedFieldsReaderClone;
        }
        return storedFieldsReader;
    }
}

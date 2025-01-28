/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
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
        // If the visitor has explicitly indicated it does not need the fields, we should not inject them
        boolean isVisitorNeedFields = true;
        if (storedFieldVisitor instanceof FieldsVisitor) {
            isVisitorNeedFields = derivedSourceVectorInjector.shouldInject(
                ((FieldsVisitor) storedFieldVisitor).includes(),
                ((FieldsVisitor) storedFieldVisitor).excludes()
            );
        }
        if (shouldInject && isVisitorNeedFields) {
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

    /**
     * For merging, we need to tell the derived source stored fields reader to skip injecting the source. Otherwise,
     * on merge we will end up just writing the source to disk
     *
     * @param storedFieldsReader stored fields reader to wrap
     * @return wrapped stored fields reader
     */
    public static StoredFieldsReader wrapForMerge(StoredFieldsReader storedFieldsReader) {
        if (storedFieldsReader instanceof DerivedSourceStoredFieldsReader) {
            StoredFieldsReader storedFieldsReaderClone = storedFieldsReader.clone();
            ((DerivedSourceStoredFieldsReader) storedFieldsReaderClone).setShouldInject(false);
            return storedFieldsReaderClone;
        }
        return storedFieldsReader;
    }
}

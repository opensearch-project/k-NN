/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.AllArgsConstructor;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.index.mapper.SourceFieldMapper;

import java.io.IOException;

/**
 * Custom {@link StoredFieldVisitor} that wraps an upstream delegate visitor in order to transparently inject derived
 * source vector fields into the document. After the source is modified, it is forwarded to the delegate.
 */
@AllArgsConstructor
public class DerivedSourceStoredFieldVisitor extends StoredFieldVisitor {

    private final StoredFieldVisitor delegate;
    private final Integer documentId;
    private final DerivedSourceVectorInjector derivedSourceVectorInjector;

    @Override
    public void binaryField(FieldInfo fieldInfo, byte[] value) throws IOException {
        // TODO: Add skip condition here if the delegate specifies which fields are not required for source
        if (fieldInfo.name.equals(SourceFieldMapper.NAME)) {
            delegate.binaryField(fieldInfo, derivedSourceVectorInjector.injectVectors(documentId, value));
            return;
        }
        delegate.binaryField(fieldInfo, value);
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) throws IOException {
        return delegate.needsField(fieldInfo);
    }
}

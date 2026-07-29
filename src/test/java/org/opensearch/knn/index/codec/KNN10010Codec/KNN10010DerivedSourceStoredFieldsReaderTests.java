/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.index.fieldvisitor.FieldsVisitor;
import org.opensearch.knn.KNNTestCase;

public class KNN10010DerivedSourceStoredFieldsReaderTests extends KNNTestCase {

    public void testResolveExcludes_usesCodecExcludesNotResponseExcludes() {
        // Response-level excludes strip the vector from the top-level _source, but codec excludes omit it
        // because an inner hit requested the field — so injection must key off codec excludes (#3303 / core #22521).
        String[] responseExcludes = { "nested_obj.vec" };
        String[] codecExcludes = new String[0];
        FieldsVisitor visitor = new FieldsVisitor(true, new String[0], responseExcludes, codecExcludes);

        assertArrayEquals(codecExcludes, KNN10010DerivedSourceStoredFieldsReader.resolveExcludes(visitor));
        assertFalse(
            "resolveExcludes must not return the response-level excludes",
            java.util.Arrays.equals(responseExcludes, KNN10010DerivedSourceStoredFieldsReader.resolveExcludes(visitor))
        );
    }

    public void testResolveExcludes_returnsCodecExcludesWhenBothPresent() {
        String[] responseExcludes = { "a", "b" };
        String[] codecExcludes = { "a" };
        FieldsVisitor visitor = new FieldsVisitor(true, new String[0], responseExcludes, codecExcludes);

        assertArrayEquals(codecExcludes, KNN10010DerivedSourceStoredFieldsReader.resolveExcludes(visitor));
    }

    public void testResolveIncludes_returnsVisitorIncludes() {
        String[] includes = { "nested_obj.vec" };
        FieldsVisitor visitor = new FieldsVisitor(true, includes, new String[0], new String[0]);

        assertArrayEquals(includes, KNN10010DerivedSourceStoredFieldsReader.resolveIncludes(visitor));
    }

    public void testResolveIncludesAndExcludes_nonFieldsVisitor_returnsNull() {
        // A visitor that is not a FieldsVisitor carries no include/exclude context — inject everything.
        StoredFieldVisitor visitor = new StoredFieldVisitor() {
            @Override
            public Status needsField(FieldInfo fieldInfo) {
                return Status.NO;
            }
        };

        assertNull(KNN10010DerivedSourceStoredFieldsReader.resolveIncludes(visitor));
        assertNull(KNN10010DerivedSourceStoredFieldsReader.resolveExcludes(visitor));
    }
}

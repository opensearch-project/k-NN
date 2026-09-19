/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.lucene.Lucene;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Shared test helpers for the prefetchable-scorer tests: reader navigation to the per-field
 * {@link FlatVectorsReader} and unwrapping a {@link PrefetchableFlatVectorScorer}. Kept in one place so the format
 * component tests (which live in different packages) do not each re-declare the same reflection boilerplate.
 */
public final class PrefetchableScorerTestUtils {

    private PrefetchableScorerTestUtils() {}

    /** Returns the scorer that a {@link PrefetchableFlatVectorScorer} delegates to. */
    public static FlatVectorsScorer getDelegate(final PrefetchableFlatVectorScorer wrapper) throws Exception {
        final Field delegate = PrefetchableFlatVectorScorer.class.getDeclaredField("delegateScorer");
        delegate.setAccessible(true);
        return (FlatVectorsScorer) delegate.get(wrapper);
    }

    /** Navigates a reader to the per-field {@link FlatVectorsReader} backing the given field (leaf 0). */
    public static FlatVectorsReader flatVectorsReaderFor(final DirectoryReader reader, final String field) throws Exception {
        final LeafReader leafReader = reader.leaves().get(0).reader();
        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);

        final KnnVectorsReader perFieldReader = segmentReader.getVectorReader();
        if (perFieldReader instanceof PerFieldKnnVectorsFormat.FieldsReader == false) {
            throw new IllegalStateException("expected a PerFieldKnnVectorsFormat.FieldsReader but was " + perFieldReader);
        }
        final KnnVectorsReader fieldReader = ((PerFieldKnnVectorsFormat.FieldsReader) perFieldReader).getFieldReader(field);
        if (fieldReader == null) {
            throw new IllegalStateException("expected a per-field reader for " + field);
        }

        // Lucene99HnswVectorsReader delegates scoring to an inner FlatVectorsReader.
        return (FlatVectorsReader) getFieldValueOfType(fieldReader, FlatVectorsReader.class);
    }

    /** Returns the value of the first declared field of {@code owner} assignable to {@code fieldType}. */
    public static Object getFieldValueOfType(final Object owner, final Class<?> fieldType) throws Exception {
        for (Field f : owner.getClass().getDeclaredFields()) {
            if (fieldType.isAssignableFrom(f.getType())) {
                f.setAccessible(true);
                return f.get(owner);
            }
        }
        throw new IllegalStateException("No field of type " + fieldType.getName() + " on " + owner.getClass().getName());
    }

    /** Returns the first declared field of {@code owner} assignable to {@code fieldType}. */
    public static Field findFieldOfType(final Class<?> owner, final Class<?> fieldType) {
        for (Field f : owner.getDeclaredFields()) {
            if (fieldType.isAssignableFrom(f.getType())) {
                return f;
            }
        }
        throw new IllegalStateException("No field of type " + fieldType.getName() + " on " + owner.getName());
    }

    /** Returns the first declared {@code static} field of {@code owner} assignable to {@code fieldType}. */
    public static Field findStaticFieldOfType(final Class<?> owner, final Class<?> fieldType) {
        for (Field f : owner.getDeclaredFields()) {
            if (Modifier.isStatic(f.getModifiers()) && fieldType.isAssignableFrom(f.getType())) {
                return f;
            }
        }
        throw new IllegalStateException("No static field of type " + fieldType.getName() + " on " + owner.getName());
    }
}

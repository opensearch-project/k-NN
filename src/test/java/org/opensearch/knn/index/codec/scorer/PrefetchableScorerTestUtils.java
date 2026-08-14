/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.junit.Assume;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.lucene.Lucene99ScorerPatcher;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Shared test helpers for the prefetchable-scorer tests: reflection over the shared {@code Lucene99FlatVectorsFormat}
 * scorer, patcher-flag reset, reader navigation to the per-field {@link FlatVectorsReader}, and unwrapping a
 * {@link PrefetchableFlatVectorScorer}. Kept in one place so the patcher test and the two format component tests
 * (which live in different packages) do not each re-declare the same reflection boilerplate.
 */
public final class PrefetchableScorerTestUtils {

    private PrefetchableScorerTestUtils() {}

    /**
     * Handle for the process-wide {@code FlatVectorsScorer} on the shared {@code Lucene99FlatVectorsFormat}, used to
     * save the pre-test scorer and restore it afterwards so tests do not pollute one another.
     */
    public static final class SharedScorerState {
        private final Object flatFormat;
        private final Field scorerField;
        private final FlatVectorsScorer originalScorer;

        private SharedScorerState(final Object flatFormat, final Field scorerField, final FlatVectorsScorer originalScorer) {
            this.flatFormat = flatFormat;
            this.scorerField = scorerField;
            this.originalScorer = originalScorer;
        }

        /** Restores the scorer captured at {@link #resolveAndReset()} time and clears the patcher flag. */
        public void restore() throws Exception {
            if (scorerField != null && flatFormat != null && originalScorer != null) {
                scorerField.set(flatFormat, originalScorer);
            }
            resetPatcherInstalledFlag();
        }
    }

    /**
     * Resolves the shared scorer, unwraps any existing prefetch wrapper so the test starts from a clean stock scorer
     * (production bootstrap or another test in the same JVM may already have wrapped it), resets the patcher flag, and
     * returns a handle for restoring later. Skips the test when the JVM forbids mutating the {@code private final}
     * scorer field, since the patch itself cannot work in that case.
     */
    public static SharedScorerState resolveAndReset() throws Exception {
        final Field fmtField = findStaticFieldOfType(Lucene99HnswVectorsFormat.class, FlatVectorsFormat.class);
        fmtField.setAccessible(true);
        final Object flatFormat = fmtField.get(null);

        final Field scorerField = findFieldOfType(flatFormat.getClass(), FlatVectorsScorer.class);
        scorerField.setAccessible(true);

        FlatVectorsScorer current = (FlatVectorsScorer) scorerField.get(flatFormat);
        if (current instanceof PrefetchableFlatVectorScorer) {
            current = getDelegate((PrefetchableFlatVectorScorer) current);
        }

        try {
            scorerField.set(flatFormat, current);
        } catch (Exception e) {
            Assume.assumeNoException("Cannot mutate Lucene99FlatVectorsFormat scorer field in this JVM", e);
        }

        resetPatcherInstalledFlag();
        return new SharedScorerState(flatFormat, scorerField, current);
    }

    /** Reflectively resets {@code Lucene99ScorerPatcher.installed} to {@code false}. */
    public static void resetPatcherInstalledFlag() throws Exception {
        final Field installed = Lucene99ScorerPatcher.class.getDeclaredField("installed");
        installed.setAccessible(true);
        installed.setBoolean(null, false);
    }

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

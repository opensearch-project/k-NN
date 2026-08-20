/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Quick-and-dirty: reflectively wrap the FlatVectorsScorer inside the stock
 * {@link Lucene99HnswVectorsFormat}'s shared {@code Lucene99FlatVectorsFormat} with a
 * {@link PrefetchableFlatVectorScorer}.
 *
 * <p>The scorer lives in a {@code private final} <em>instance</em> field ({@code vectorsScorer}) on the
 * single shared {@code Lucene99FlatVectorsFormat} that {@code Lucene99HnswVectorsFormat} holds in a
 * {@code private static final} field. We read that shared instance and overwrite its scorer field once.
 * Because {@code Lucene99FlatVectorsFormat#fieldsReader/fieldsWriter} read the field at call time, every
 * reader/writer created after this runs picks up the wrapped scorer. The on-disk format name is unchanged,
 * so there is no backward-compatibility impact.
 *
 * <p>Call {@link #installOnce()} once at plugin startup, before any index open/write.
 *
 * <p><b>Caveats:</b> this is a node-wide monkeypatch affecting every stock Lucene99 HNSW field (read and
 * write); it must run before any index activity; it may require {@code suppressAccessChecks} /
 * {@code accessDeclaredMembers} reflection permissions under a SecurityManager; and it is fragile across
 * Lucene upgrades since it relies on {@code Lucene99FlatVectorsFormat} having exactly one
 * {@link FlatVectorsScorer} field.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class Lucene99ScorerPatcher {

    private static volatile boolean installed = false;

    /**
     * Idempotently wraps the stock Lucene99 flat vector scorer with a {@link PrefetchableFlatVectorScorer}.
     * Safe to call multiple times; only the first call mutates state.
     */
    public static synchronized void installOnce() {
        if (installed) {
            return;
        }
        try {
            // 1) Read the shared Lucene99FlatVectorsFormat instance held in Lucene99HnswVectorsFormat's
            // private static final field. Referencing the class forces its static init first.
            final Field fmtField = findStaticFieldOfType(Lucene99HnswVectorsFormat.class, FlatVectorsFormat.class);
            fmtField.setAccessible(true);
            final Object flatFormat = fmtField.get(null); // static field -> null receiver

            // 2) Locate the private final FlatVectorsScorer field on the flat format instance.
            final Field scorerField = findFieldOfType(flatFormat.getClass(), FlatVectorsScorer.class);
            scorerField.setAccessible(true);

            final FlatVectorsScorer current = (FlatVectorsScorer) scorerField.get(flatFormat);
            if (current instanceof PrefetchableFlatVectorScorer) {
                installed = true; // already patched
                return;
            }

            // 3) Overwrite the instance field with the prefetching wrapper.
            scorerField.set(flatFormat, new PrefetchableFlatVectorScorer(current));
            installed = true;
        } catch (Exception e) {
            log.warn("Failed to patch Lucene99 flat vector scorer to add Prefetchable Vector Scorer", e);
        }
    }

    private static Field findStaticFieldOfType(final Class<?> owner, final Class<?> fieldType) {
        for (Field f : owner.getDeclaredFields()) {
            if (Modifier.isStatic(f.getModifiers()) && fieldType.isAssignableFrom(f.getType())) {
                return f;
            }
        }
        throw new IllegalStateException("No static field of type " + fieldType.getName() + " on " + owner.getName());
    }

    private static Field findFieldOfType(final Class<?> owner, final Class<?> fieldType) {
        for (Field f : owner.getDeclaredFields()) {
            if (fieldType.isAssignableFrom(f.getType())) {
                return f;
            }
        }
        throw new IllegalStateException("No field of type " + fieldType.getName() + " on " + owner.getName());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import java.io.Closeable;
import java.util.Arrays;
import java.util.Collection;
import org.apache.lucene.store.Directory;

/**
 * Minimal KNNIOUtils for resource and file cleanup.
 */
public class KNNIOUtils {
    private KNNIOUtils() {}

    /**
     * Closes all given Closeables, suppressing all thrown Throwables in ex.
     * Some of the Closeables may be null, they are ignored.
     */
    public static void closeWhileSuppressingExceptions(Throwable ex, Closeable... objects) {
        Error firstError = ex instanceof Error err ? err : null;
        for (Closeable object : objects) {
            try {
                if (object != null) {
                    object.close();
                }
            } catch (Throwable e) {
                if (firstError == null && e instanceof Error err) {
                    firstError = err;
                    firstError.addSuppressed(ex);
                } else {
                    ex.addSuppressed(e);
                }
            }
        }

        if (firstError != null) {
            throw firstError;
        }
    }

    /**
     * Deletes all given files, suppressing all thrown Throwables in ex.
     * Note that the files should not be null.
     */
    public static void deleteFilesSuppressingExceptions(Throwable ex, Directory dir, Collection<String> files) {
        Error firstError = ex instanceof Error err ? err : null;
        for (String name : files) {
            try {
                dir.deleteFile(name);
            } catch (Throwable d) {
                if (firstError == null && d instanceof Error err) {
                    firstError = err;
                    firstError.addSuppressed(ex);
                } else {
                    ex.addSuppressed(d);
                }
            }
        }
        if (firstError != null) {
            throw firstError;
        }
    }

    public static void deleteFilesSuppressingExceptions(Throwable ex, Directory dir, String... files) {
        deleteFilesSuppressingExceptions(ex, dir, Arrays.asList(files));
    }
}

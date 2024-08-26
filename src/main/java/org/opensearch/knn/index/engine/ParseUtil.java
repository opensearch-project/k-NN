/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import java.util.Objects;

public final class ParseUtil {
    public static String unwrapString(String in, char expectedStart, char expectedEnd) {
        if (in.length() < 2) {
            throw new IllegalArgumentException("Invalid string.");
        }

        if (in.charAt(0) != expectedStart || in.charAt(in.length() - 1) != expectedEnd) {
            throw new IllegalArgumentException("Invalid string." + in);
        }
        return in.substring(1, in.length() - 1);
    }

    public static int findClosingPosition(String in, char expectedStart, char expectedEnd) {
        int nestedLevel = 0;
        for (int i = 0; i < in.length(); i++) {
            if (in.charAt(i) == expectedStart) {
                nestedLevel++;
                continue;
            }

            if (in.charAt(i) == expectedEnd) {
                nestedLevel--;
            }

            if (nestedLevel == 0) {
                return i;
            }
        }

        throw new IllegalArgumentException("Invalid string. No end to the nesting");
    }

    public static void checkStringNotEmpty(String string) {
        if (string.isEmpty()) {
            throw new IllegalArgumentException("Unable to parse MethodComponentContext");
        }
    }

    public static void checkStringMatches(String string, String expected) {
        if (!Objects.equals(string, expected)) {
            throw new IllegalArgumentException("Unexpected key in MethodComponentContext.  Expected 'name' or 'parameters'");
        }
    }

    public static void checkExpectedArrayLength(String[] array, int expectedLength) {
        if (null == array) {
            throw new IllegalArgumentException("Error parsing MethodComponentContext.  Array is null.");
        }

        if (array.length != expectedLength) {
            throw new IllegalArgumentException("Error parsing MethodComponentContext.  Array is not expected length.");
        }
    }
}

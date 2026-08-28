/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import org.opensearch.test.OpenSearchTestCase;

import java.lang.annotation.ElementType;
import java.lang.annotation.RetentionPolicy;

/**
 * Tests for {@link ExperimentalAlgorithm} marker annotation.
 */
public class ExperimentalAlgorithmTests extends OpenSearchTestCase {

    @ExperimentalAlgorithm(description = "Test algorithm for validation", since = "3.6.0")
    private static class SampleExperimentalAlgorithm {
        // Intentionally empty — used only to verify annotation behavior
    }

    @ExperimentalAlgorithm
    private static class MinimalExperimentalAlgorithm {
        // Intentionally empty — verifies default annotation values
    }

    public void testAnnotationIsPresent() {
        assertTrue(
            "SampleExperimentalAlgorithm should be annotated with @ExperimentalAlgorithm",
            SampleExperimentalAlgorithm.class.isAnnotationPresent(ExperimentalAlgorithm.class)
        );
    }

    public void testAnnotationDescription() {
        ExperimentalAlgorithm annotation = SampleExperimentalAlgorithm.class.getAnnotation(ExperimentalAlgorithm.class);
        assertNotNull("Annotation should not be null", annotation);
        assertEquals("Test algorithm for validation", annotation.description());
    }

    public void testAnnotationSince() {
        ExperimentalAlgorithm annotation = SampleExperimentalAlgorithm.class.getAnnotation(ExperimentalAlgorithm.class);
        assertNotNull("Annotation should not be null", annotation);
        assertEquals("3.6.0", annotation.since());
    }

    public void testAnnotationDefaultValues() {
        ExperimentalAlgorithm annotation = MinimalExperimentalAlgorithm.class.getAnnotation(ExperimentalAlgorithm.class);
        assertNotNull("Annotation should not be null", annotation);
        assertEquals("Default description should be empty", "", annotation.description());
        assertEquals("Default since should be empty", "", annotation.since());
    }

    public void testAnnotationRetentionPolicy() {
        java.lang.annotation.Retention retention = ExperimentalAlgorithm.class.getAnnotation(java.lang.annotation.Retention.class);
        assertNotNull("@Retention should be present", retention);
        assertEquals("Retention should be RUNTIME", RetentionPolicy.RUNTIME, retention.value());
    }

    public void testAnnotationTargetType() {
        java.lang.annotation.Target target = ExperimentalAlgorithm.class.getAnnotation(java.lang.annotation.Target.class);
        assertNotNull("@Target should be present", target);
        assertEquals("Should have exactly one target", 1, target.value().length);
        assertEquals("Target should be TYPE", ElementType.TYPE, target.value()[0]);
    }
}

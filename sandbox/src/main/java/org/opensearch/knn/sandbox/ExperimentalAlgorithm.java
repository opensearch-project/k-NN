/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marker annotation indicating that a class is an experimental algorithm or component
 * within the k-NN sandbox module.
 *
 * <p>Classes annotated with {@code @ExperimentalAlgorithm} are not part of the stable,
 * production-ready API surface. They may change, be renamed, or be removed without
 * prior notice in any future release.</p>
 *
 * <p>Usage example:</p>
 * <pre>{@code
 * @ExperimentalAlgorithm(
 *     description = "Graph-based approximate nearest neighbor using vamana algorithm",
 *     since = "3.6.0"
 * )
 * public class VamanaAlgorithm {
 *     // experimental implementation
 * }
 * }</pre>
 *
 * <p><strong>Graduation Criteria</strong></p>
 * <p>An experimental component may graduate to the main package when it meets:</p>
 * <ul>
 *   <li>Stability: No critical bugs after a minimum incubation period</li>
 *   <li>Performance: Meets or exceeds benchmarks comparable to existing stable algorithms</li>
 *   <li>Test coverage: Comprehensive unit and integration tests</li>
 *   <li>Community feedback: Positive reception from early adopters</li>
 * </ul>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.TYPE })
public @interface ExperimentalAlgorithm {

    /**
     * A brief description of the experimental algorithm or component.
     *
     * @return the description string, empty by default
     */
    String description() default "";

    /**
     * The version in which this experimental component was first introduced.
     *
     * @return the version string (e.g., "3.6.0"), empty by default
     */
    String since() default "";
}

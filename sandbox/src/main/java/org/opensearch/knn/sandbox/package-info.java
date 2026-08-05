/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * The {@code org.opensearch.knn.sandbox} package provides a first-class incubation
 * environment for experimental algorithms, optimizations, and architectural components
 * within the OpenSearch k-NN plugin.
 *
 * <p>All classes in this package and its sub-packages are considered <strong>experimental</strong>
 * and are not included in production release artifacts. They follow a defined lifecycle:</p>
 * <ol>
 *   <li>New ideas land in this sandbox module.</li>
 *   <li>They are exposed to early adopters and community feedback under an explicit experimental label.</li>
 *   <li>Once they meet stability, performance, and test coverage thresholds, they graduate to the main package.</li>
 * </ol>
 *
 * <p>Every public class in this package should be annotated with
 * {@link org.opensearch.knn.sandbox.ExperimentalAlgorithm @ExperimentalAlgorithm}.</p>
 *
 * @see org.opensearch.knn.sandbox.ExperimentalAlgorithm
 */
package org.opensearch.knn.sandbox;

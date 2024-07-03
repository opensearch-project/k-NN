/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.util;

import org.opensearch.knn.index.Parameter;

import java.util.Collections;
import java.util.Map;

/**
 * Holds context related to a method for a particular engine
 * Each engine can have a specific set of parameters that it supports during index and build time. This context holds
 * the information for each engine method combination.
 *
 * TODO: Move KnnMethod in here
 */
public interface EngineSpecificMethodContext {

    Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx);

    EngineSpecificMethodContext EMPTY = ctx -> Collections.emptyMap();
}

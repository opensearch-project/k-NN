/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

/**
 * This interface is used to indicate if derived source is enabled for a field type
 */
public interface DerivedSourceFieldType {

    /**
     *
     * @return boolean Returns true if derived source is enabled for this field
     */
    boolean isDerivedSourceEnabled();
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.opensearch.common.ValidationException;

@Slf4j
@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class RescoreContext {

    public static final float DEFAULT_OVERSAMPLE_FACTOR = 1.0f;
    public static final float MAX_OVERSAMPLE_FACTOR = 5.0f;
    public static final float MIN_OVERSAMPLE_FACTOR = 0.0f;

    @Builder.Default
    private float oversampleFactor = DEFAULT_OVERSAMPLE_FACTOR;

    /**
     * Validate the rescore context
     *
     * @return ValidationException if validation fails, null otherwise
     */
    public ValidationException validate() {
        if (oversampleFactor < RescoreContext.MIN_OVERSAMPLE_FACTOR) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    "Oversample factor [%f] cannot be less than [%f]",
                    getOversampleFactor(),
                    RescoreContext.MIN_OVERSAMPLE_FACTOR
                )
            );
            return validationException;
        }

        if (oversampleFactor > RescoreContext.MAX_OVERSAMPLE_FACTOR) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    "Oversample factor [%f] cannot be more than [%f]",
                    getOversampleFactor(),
                    RescoreContext.MAX_OVERSAMPLE_FACTOR
                )
            );
            return validationException;
        }
        return null;
    }

    /**
     *
     * @return default RescoreContext
     */
    public static RescoreContext getDefault() {
        return RescoreContext.builder().build();
    }
}

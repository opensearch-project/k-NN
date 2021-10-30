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

package org.opensearch.knn.yamlRestTest;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.opensearch.test.rest.yaml.ClientYamlTestCandidate;
import org.opensearch.test.rest.yaml.OpenSearchClientYamlSuiteTestCase;


public class KNNClientYamlTestSuiteIT extends OpenSearchClientYamlSuiteTestCase {

  public KNNClientYamlTestSuiteIT(@Name("yaml") ClientYamlTestCandidate testCandidate) {
    super(testCandidate);
  }

  @ParametersFactory
    public static Iterable<Object[]> parameters() throws Exception {
    return OpenSearchClientYamlSuiteTestCase.createParameters();
  }
}
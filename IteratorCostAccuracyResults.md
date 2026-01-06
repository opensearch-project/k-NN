re are the results:

| Iterator Type | Cardinality | cost() | actual | accuracy | exact? |
|--------------|-------------|--------|--------|----------|--------|
| **TermQuery** | Low (5) | 5 | 5 | 1.00 | ✅ |
| **TermQuery** | Medium (10) | 10 | 10 | 1.00 | ✅ |
| **TermQuery** | High (1000) | 1000 | 1000 | 1.00 | ✅ |
| **BooleanMUST** | Low (5) | 5 | 5 | 1.00 | ✅ |
| **BooleanMUST** | Medium (10) | 10 | 10 | 1.00 | ✅ |
| **BooleanSHOULD** | Low (5) | 5 | 5 | 1.00 | ✅ |
| **BooleanSHOULD** | High | 15 | 10 | 1.50 | ❌ (overestimate) |
| **PointRangeQuery** | Low (5) | 385 | 5 | 77.00 | ❌ (huge overestimate) |
| **PointRangeQuery** | Medium (10) | 385 | 10 | 38.50 | ❌ (huge overestimate) |
| **PointRangeQuery** | High (1000) | 1155 | 1000 | 1.16 | ❌ (slight overestimate) |
| **BitSetIterator** | All | exact | exact | 1.00 | ✅ |
| **Mixed (Term+Point)** | 10 | 385 | 10 | 38.50 | ❌ (PointRange dominates) |

**Key findings:**
- **TermQuery**: Always exact - safe to use `cost()`
- **BooleanMUST (AND)**: Exact when sub-queries are exact
- **BooleanSHOULD (OR)**: Sums costs, can overestimate with overlapping docs
- **PointRangeQuery**: Significantly overestimates, especially at low cardinality (BKD tree estimation)
- **BitSetIterator**: Always exact (cardinality passed at construction)

Full test results: 
./gradlew :test --tests "org.opensearch.knn.index.query.IteratorCostAccuracyTests" --info | pbcopy

IteratorCostAccuracyTests > testMixedQueryCost_TermAndPointRange STANDARD_OUT
    [2026-01-06T12:36:59,162][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testMixedQueryCost_TermAndPointRange] before test
    [2026-01-06T12:36:59,203][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testMixedQueryCost_TermAndPointRange] Mixed-TermAndPoint: cost=512, actual=10, accuracy=51.20, exact=false, overestimate=true
    [2026-01-06T12:36:59,213][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testMixedQueryCost_TermAndPointRange] after test

IteratorCostAccuracyTests > testMixedQueryCost_TermAndPointRange STANDARD_ERROR
    SLF4J(W): No SLF4J providers were found.
    SLF4J(W): Defaulting to no-operation (NOP) logger implementation
    SLF4J(W): See https://www.slf4j.org/codes.html#noProviders for further details.

IteratorCostAccuracyTests > testMixedQueryCost_TermAndPointRange STANDARD_OUT
    [2026-01-06T12:36:59,244][WARN ][o.o.k.i.m.NativeMemoryCacheManager] [testMixedQueryCost_TermAndPointRange] ThreadPool is null during NativeMemoryCacheManager initialization. Maintenance will not start.
    [2026-01-06T12:36:59,246][WARN ][o.o.k.q.m.q.QuantizationStateCache] [testMixedQueryCost_TermAndPointRange] ThreadPool is null during QuantizationStateCache initialization. Maintenance will not start.

IteratorCostAccuracyTests > testPointRangeQueryCost_MediumCardinality STANDARD_OUT
    [2026-01-06T12:36:59,336][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_MediumCardinality] before test
    [2026-01-06T12:36:59,337][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_MediumCardinality] PointRange-Medium: cost=512, actual=10, accuracy=51.20, exact=false, overestimate=true
    [2026-01-06T12:36:59,337][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_MediumCardinality] after test

IteratorCostAccuracyTests > testBooleanShouldQueryCost_LowCardinality STANDARD_OUT
    [2026-01-06T12:36:59,411][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_LowCardinality] before test
    [2026-01-06T12:36:59,414][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_LowCardinality] BooleanSHOULD-Low: cost=5, actual=5, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,415][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_LowCardinality] after test

IteratorCostAccuracyTests > testTermQueryCost_LowCardinality STANDARD_OUT
    [2026-01-06T12:36:59,463][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_LowCardinality] before test
    [2026-01-06T12:36:59,463][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_LowCardinality] TermQuery-Low: cost=5, actual=5, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,463][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_LowCardinality] after test

IteratorCostAccuracyTests > testBooleanMustQueryCost_MediumCardinality STANDARD_OUT
    [2026-01-06T12:36:59,504][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_MediumCardinality] before test
    [2026-01-06T12:36:59,504][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_MediumCardinality] BooleanMUST-Medium: cost=10, actual=10, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,505][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_MediumCardinality] after test

IteratorCostAccuracyTests > testBooleanShouldQueryCost_HighCardinality STANDARD_OUT
    [2026-01-06T12:36:59,540][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_HighCardinality] before test
    [2026-01-06T12:36:59,543][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_HighCardinality] BooleanSHOULD-High: cost=15, actual=10, accuracy=1.50, exact=false, overestimate=true
    [2026-01-06T12:36:59,543][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanShouldQueryCost_HighCardinality] after test

IteratorCostAccuracyTests > testBitSetIteratorCost_MediumCardinality STANDARD_OUT
    [2026-01-06T12:36:59,574][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_MediumCardinality] before test
    [2026-01-06T12:36:59,574][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_MediumCardinality] BitSetIterator-Medium: cost=10, actual=10, accuracy=1.00, exact=true
    [2026-01-06T12:36:59,574][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_MediumCardinality] after test

IteratorCostAccuracyTests > testBitSetIteratorCost_LowCardinality STANDARD_OUT
    [2026-01-06T12:36:59,602][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_LowCardinality] before test
    [2026-01-06T12:36:59,603][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_LowCardinality] BitSetIterator-Low: cost=5, actual=5, accuracy=1.00, exact=true
    [2026-01-06T12:36:59,603][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_LowCardinality] after test

IteratorCostAccuracyTests > testPointRangeQueryCost_LowCardinality STANDARD_OUT
    [2026-01-06T12:36:59,629][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_LowCardinality] before test
    [2026-01-06T12:36:59,629][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_LowCardinality] PointRange-Low: cost=512, actual=5, accuracy=102.40, exact=false, overestimate=true
    [2026-01-06T12:36:59,630][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_LowCardinality] after test

IteratorCostAccuracyTests > testBitSetIteratorCost_HighCardinality STANDARD_OUT
    [2026-01-06T12:36:59,653][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_HighCardinality] before test
    [2026-01-06T12:36:59,653][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_HighCardinality] BitSetIterator-High: cost=1000, actual=1000, accuracy=1.00, exact=true
    [2026-01-06T12:36:59,653][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBitSetIteratorCost_HighCardinality] after test

IteratorCostAccuracyTests > testBooleanMustQueryCost_LowCardinality STANDARD_OUT
    [2026-01-06T12:36:59,675][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_LowCardinality] before test
    [2026-01-06T12:36:59,678][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_LowCardinality] BooleanMUST-Low: cost=5, actual=5, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,678][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testBooleanMustQueryCost_LowCardinality] after test

IteratorCostAccuracyTests > testPointRangeQueryCost_HighCardinality STANDARD_OUT
    [2026-01-06T12:36:59,704][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_HighCardinality] before test
    [2026-01-06T12:36:59,704][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_HighCardinality] PointRange-High: cost=1024, actual=1000, accuracy=1.02, exact=false, overestimate=true
    [2026-01-06T12:36:59,705][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testPointRangeQueryCost_HighCardinality] after test

IteratorCostAccuracyTests > testTermQueryCost_MediumCardinality STANDARD_OUT
    [2026-01-06T12:36:59,726][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_MediumCardinality] before test
    [2026-01-06T12:36:59,727][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_MediumCardinality] TermQuery-Medium: cost=10, actual=10, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,727][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_MediumCardinality] after test

IteratorCostAccuracyTests > testTermQueryCost_HighCardinality STANDARD_OUT
    [2026-01-06T12:36:59,748][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_HighCardinality] before test
    [2026-01-06T12:36:59,749][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_HighCardinality] TermQuery-High: cost=1000, actual=1000, accuracy=1.00, exact=true, overestimate=true
    [2026-01-06T12:36:59,749][INFO ][o.o.k.i.q.IteratorCostAccuracyTests] [testTermQueryCost_HighCardinality] after test

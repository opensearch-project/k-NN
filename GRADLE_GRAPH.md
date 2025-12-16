# Gradle Build Task Graph

Dependency DAG with transitive dependencies removed for clarity.

Prompt AI with questions like 'Analyzing GRADLE_GRAPH.md, does buildJniTest come before integTest?'

Regenerate with `./gradlew generateTaskGraph`

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 50, "rankSpacing": 80}}}%%
graph TD
    check --> build
    assemble --> build
    spotlessCheck --> check
    javadoc --> check
    jacocoTestReport --> check
    spotlessJavaCheck --> spotlessCheck
    spotlessJavaApply --> spotlessJavaCheck
    spotlessInternalRegisterDependencies --> spotlessJava
    clean --> spotlessJava
    spotlessJava --> spotlessJavaApply
    delombok --> javadoc
    generateEffectiveLombokConfig --> compileJava
    compileJava --> delombok
    integTest --> jacocoTestReport
    compileJava --> classes
    processResources --> classes
    bundlePlugin --> integTest
    test --> integTest
    compileTestFixturesJava --> compileTestJava
    generateTestEffectiveLombokConfig --> compileTestJava
    compileJava --> compileTestFixturesJava
    generateTestFixturesEffectiveLombokConfig --> compileTestFixturesJava
    pluginProperties --> bundlePlugin
    jar --> bundlePlugin
    generateNotice --> bundlePlugin
    copyPluginPropertiesTemplate --> pluginProperties
    classes --> jar
    processTestResources --> testClasses
    compileTestJava --> testClasses
    pluginProperties --> testClasses
    testFixturesClasses --> testFixturesJar
    compileTestFixturesJava --> testFixturesClasses
    processTestFixturesResources --> testFixturesClasses
    cmakeJniLib --> buildJniLib
    buildJniTest --> test
    buildJniLib --> buildJniTest
    prepareJavaAgent --> buildJniTest
    precommit --> buildJniTest
    thirdPartyAudit --> precommit
    forbiddenPatterns --> precommit
    jarHell --> precommit
    forbiddenApis --> precommit
    licenseHeaders --> precommit
    testingConventions --> precommit
    validatePom --> precommit
    filepermissions --> precommit
    dependencyLicenses --> precommit
    loggerUsageCheck --> precommit
    thirdPartyAuditResources --> thirdPartyAudit
    testClasses --> jarHell
    testFixturesJar --> jarHell
    jar --> jarHell
    forbiddenApisTestFixtures --> forbiddenApis
    forbiddenApisTest --> forbiddenApis
    forbiddenApisMain --> forbiddenApis
    testFixturesClasses --> forbiddenApisTestFixtures
    jar --> forbiddenApisTestFixtures
    forbiddenApisResources --> forbiddenApisTestFixtures
    testClasses --> forbiddenApisTest
    testFixturesJar --> forbiddenApisTest
    jar --> forbiddenApisTest
    forbiddenApisResources --> forbiddenApisTest
    jar --> forbiddenApisMain
    forbiddenApisResources --> forbiddenApisMain
    testClasses --> testingConventions
    testFixturesJar --> testingConventions
    jar --> testingConventions
    validateNebulaPom --> validatePom
    validatePluginZipPom --> validatePom
    generatePomFileForNebulaPublication --> validateNebulaPom
    generatePomFileForPluginZipPublication --> validateNebulaPom
    generatePomFileForPluginZipPublication --> validatePluginZipPom
    generatePomFileForNebulaPublication --> validatePluginZipPom
    compileTestJava --> loggerUsageCheck
    javadocJar --> assemble
    sourcesJar --> assemble
    bundlePlugin --> assemble
    generatePom --> assemble
    javadoc --> javadocJar
    generatePomFileForPluginZipPublication --> generatePom
    generatePomFileForNebulaPublication --> generatePom
```

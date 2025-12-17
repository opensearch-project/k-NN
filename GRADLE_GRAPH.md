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
    jacocoTestReport --> check
    javadoc --> check
    spotlessJavaCheck --> spotlessCheck
    spotlessJavaApply --> spotlessJavaCheck
    spotlessInternalRegisterDependencies --> spotlessJava
    clean --> spotlessJava
    spotlessJava --> spotlessJavaApply
    integTest --> jacocoTestReport
    compileJava --> classes
    processResources --> classes
    generateEffectiveLombokConfig --> compileJava
    buildJniTest --> test
    compileTestFixturesJava --> compileTestJava
    generateTestEffectiveLombokConfig --> compileTestJava
    compileJava --> compileTestFixturesJava
    generateTestFixturesEffectiveLombokConfig --> compileTestFixturesJava
    copyPluginPropertiesTemplate --> pluginProperties
    pluginProperties --> testClasses
    processTestResources --> testClasses
    compileTestJava --> testClasses
    testFixturesClasses --> testFixturesJar
    compileTestFixturesJava --> testFixturesClasses
    processTestFixturesResources --> testFixturesClasses
    classes --> jar
    compileJava --> validateLibraryUsage
    buildJniLib --> buildJniTest
    prepareJavaAgent --> buildJniTest
    precommit --> buildJniTest
    cmakeJniLib --> buildJniLib
    dependencyLicenses --> precommit
    validatePom --> precommit
    licenseHeaders --> precommit
    forbiddenApis --> precommit
    thirdPartyAudit --> precommit
    filepermissions --> precommit
    testingConventions --> precommit
    forbiddenPatterns --> precommit
    loggerUsageCheck --> precommit
    jarHell --> precommit
    validatePluginZipPom --> validatePom
    validateNebulaPom --> validatePom
    generatePomFileForPluginZipPublication --> validatePluginZipPom
    generatePomFileForNebulaPublication --> validatePluginZipPom
    generatePomFileForNebulaPublication --> validateNebulaPom
    generatePomFileForPluginZipPublication --> validateNebulaPom
    forbiddenApisTest --> forbiddenApis
    forbiddenApisTestFixtures --> forbiddenApis
    forbiddenApisMain --> forbiddenApis
    testClasses --> forbiddenApisTest
    testFixturesJar --> forbiddenApisTest
    jar --> forbiddenApisTest
    forbiddenApisResources --> forbiddenApisTest
    testFixturesClasses --> forbiddenApisTestFixtures
    jar --> forbiddenApisTestFixtures
    forbiddenApisResources --> forbiddenApisTestFixtures
    jar --> forbiddenApisMain
    forbiddenApisResources --> forbiddenApisMain
    thirdPartyAuditResources --> thirdPartyAudit
    testClasses --> testingConventions
    testFixturesJar --> testingConventions
    jar --> testingConventions
    compileTestJava --> loggerUsageCheck
    testClasses --> jarHell
    testFixturesJar --> jarHell
    jar --> jarHell
    bundlePlugin --> integTest
    test --> integTest
    pluginProperties --> bundlePlugin
    jar --> bundlePlugin
    generateNotice --> bundlePlugin
    delombok --> javadoc
    compileJava --> delombok
    bundlePlugin --> assemble
    generatePom --> assemble
    sourcesJar --> assemble
    javadocJar --> assemble
    generatePomFileForPluginZipPublication --> generatePom
    generatePomFileForNebulaPublication --> generatePom
    javadoc --> javadocJar
    jar ==> validateLibraryUsage
```

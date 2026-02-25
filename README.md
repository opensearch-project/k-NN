[![Build and Test k-NN](https://github.com/opensearch-project/k-NN/actions/workflows/CI.yml/badge.svg)](https://github.com/opensearch-project/k-NN/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/opensearch-project/k-NN/branch/main/graph/badge.svg?token=PYQO2GW39S)](https://codecov.io/gh/opensearch-project/k-NN)
[![Documentation](https://img.shields.io/badge/doc-reference-blue)](https://opensearch.org/docs/search-plugins/knn/index/)
[![Chat](https://img.shields.io/badge/chat-on%20forums-blue)](https://forum.opensearch.org/c/plugins/k-nn/48)
![PRs welcome!](https://img.shields.io/badge/PRs-welcome!-success)

# OpenSearch k-NN
- [Welcome!](#welcome)
- [Project Resources](#project-resources)
- [Credits and  Acknowledgments](#credits-and-acknowledgments)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Copyright](#copyright)

## Welcome!

**OpenSearch k-NN** enables you to run the nearest neighbor search on billions of documents across thousands of dimensions with the same ease as running any regular OpenSearch query. You can use aggregations and filter clauses to further refine your similarity search operations. k-NN similarity search powers use cases such as product recommendations, fraud detection, image and video search, related document search, and more.

## Project Resources

* [Project Website](https://opensearch.org/)
* [Downloads](https://opensearch.org/downloads.html)
* [Documentation](https://opensearch.org/docs/search-plugins/knn/index/)
* Need help? Try the [Forum](https://forum.opensearch.org/c/plugins/k-nn/48)
* [Project Principles](https://opensearch.org/#principles)
* [Contributing to OpenSearch k-NN](CONTRIBUTING.md)
* [Maintainer Responsibilities](MAINTAINERS.md)
* [Release Management](RELEASING.md)
* [Admin Responsibilities](ADMINS.md)
* [Security](SECURITY.md)

## Credits and Acknowledgments

This project uses two similarity search libraries to perform Approximate Nearest Neighbor Search: the Apache 2.0-licensed [Non-Metric Space Library](https://github.com/nmslib/nmslib/) and the MIT licensed [Faiss library](https://github.com/facebookresearch/faiss). Thank you to all who have contributed to those projects including Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov and David Novak for nmslib and Hervé Jégou, Matthijs Douze, Jeff Johnson and Lucas Hosseini for Faiss.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](CODE_OF_CONDUCT.md). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq), or contact [opensource-codeofconduct@amazon.com](mailto:opensource-codeofconduct@amazon.com) with any additional questions or comments.

## License

This project is licensed under the [Apache v2.0 License](LICENSE.txt).

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](NOTICE.txt) for details.

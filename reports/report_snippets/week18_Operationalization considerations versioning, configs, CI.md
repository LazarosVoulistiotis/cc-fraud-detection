## Operationalization Considerations: Versioning, Configs, and CI

To reduce deployment fragility and make the API more operationally robust, the serving layer was designed with explicit versioning, configuration-based controls, structured logging, and automated testing.

### Versioning and Metadata
A dedicated `/metadata` endpoint was implemented to expose serving-time metadata for the currently deployed model. This includes the model version, resolved artifact path, threshold policy and version, feature schema version, Git commit identifier, training date, and selected training references. This endpoint acts as a lightweight model-registry view and improves traceability of the served artefacts.

### Configuration-Driven Serving
Thresholding behavior is not hardcoded in application logic. Instead, the serving decision policy is loaded from configuration files (`threshold.json`, `feature_schema.json`, and `model_metadata.json`). This separates model-serving behavior from code changes, improves reproducibility, and makes threshold or schema updates easier to manage in a controlled way.

### Reproducible Preprocessing
The API applies deterministic preprocessing before inference, including engineered feature generation and strict feature alignment to the frozen model feature order. This reduces the risk of feature-name mismatch and ensures that prediction-time inputs remain consistent with the training-time schema.

### Structured Logging and Observability
The API emits structured JSON logs for request-level and prediction-level events. Logged information includes request latency, status code, prediction latency, fraud probability, threshold used, selected label, and error events. This improves observability, debugging, and operational transparency without logging full raw payloads.

### Automated Testing
A dedicated test suite was created for both preprocessing and API behavior. Unit tests validate engineered feature creation, schema alignment, and invalid input handling, while API tests verify endpoint availability, prediction response structure, validation failures, and metadata exposure. These tests provide confidence that the core serving path behaves predictably.

### Continuous Integration
A GitHub Actions workflow was added to automatically run the test suite on each push and pull request to the main branch. This provides a lightweight CI gate, ensuring that breaking changes in preprocessing, serving logic, or API contracts are detected early before integration.
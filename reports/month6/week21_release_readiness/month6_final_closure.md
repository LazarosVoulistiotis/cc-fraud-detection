## Month 6 — Deployment Hardening, Live Validation, and Final Project Closure

Month 6 marked the transition from a locally validated machine learning prototype to a deployment-oriented fraud detection system supported by a hardened API, containerized runtime, live cloud deployment, and final validation evidence. Across Weeks 17 to 21, the project moved beyond model development alone and focused on serving reproducibility, interface stability, deployment realism, operational interpretation, and release readiness.

### Week 17 — Frozen serving layer and deployment-oriented API design

Week 17 established the frozen inference boundary of the project. At this stage, the selected XGBoost champion model was packaged into a serving-oriented API workflow, with the final model artifact, threshold configuration, and serving schema separated from experimentation code. This step was important because it transformed the project from a modelling pipeline into a system that could be invoked predictively through clearly defined endpoints.

The API was designed to expose a minimal but practical interface, including health checking, metadata inspection, direct prediction from raw transaction features, and deterministic demo prediction through an ID-based lookup endpoint. This freeze established the serving contract that was later validated and operationalized throughout the remaining weeks.

### Week 18 — API hardening, reproducible preprocessing, logging, and automated testing

Week 18 focused on strengthening the serving layer so that inference-time behaviour would be deterministic, auditable, and reproducible. The preprocessing logic was hardened to ensure strict validation of incoming payloads, deterministic feature engineering, and exact alignment with the frozen serving schema. This reduced the risk of inference-time feature mismatch or silent schema drift.

The API was also extended with structured logging, centralized error handling, and richer metadata exposure. In addition, automated tests were added with `pytest`, and continuous integration was introduced through GitHub Actions. This week significantly improved the engineering maturity of the system by showing that the deployed inference layer was not merely functional, but also testable, traceable, and maintainable.

### Week 19 — Dockerization and runtime packaging

Week 19 moved the project from a Python-only local execution path to a portable containerized runtime. The FastAPI service was packaged with Docker using a reproducible runtime image that included the frozen model artifact, serving configuration files, and supporting demo data required by the application. A `.dockerignore` file and lightweight command structure were also introduced to improve packaging quality and usability.

This step was particularly important because it made the serving system environment-independent and closer to real deployment practice. The local smoke tests confirmed that the API behaved correctly from inside the container, demonstrating that the application, dependencies, configuration, and model artifacts were packaged coherently.

### Week 20 — Live cloud deployment and deployment architecture closure

Week 20 completed the transition to a publicly reachable deployed service through Google Cloud Run. The Dockerized application was built, pushed, and deployed successfully, producing a live HTTPS endpoint and a working Swagger/OpenAPI interface. The deployment was validated through real endpoint checks, including health, metadata, and scoring requests.

This week also formalized the deployed architecture narrative of the project by clarifying the role of the public endpoint, the API serving layer, preprocessing logic, frozen XGBoost model, and supporting monitoring concepts. As a result, the project was no longer only a simulation of deployment readiness; it had become an actual live cloud-hosted inference service.

### Week 21 — Final validation, API stress testing, and release readiness

Week 21 served as the final validation and closure phase of the deployed system. The frozen artifact, schema, and threshold configuration were verified successfully, and the locked hold-out evaluation reproduced the expected frozen-system results exactly. A post-hoc threshold comparison showed that the final serving threshold `0.1279` was more suitable than the historical `0.0884` reference because it preserved fraud recall while reducing false positives, improving precision, and slightly lowering expected cost.

This final week also distinguished clearly between the project’s early recall-first research aspiration and its final business-aware serving objective. Although the final serving threshold did not preserve Recall ≥ 0.90, it provided a much more realistic operational balance between fraud capture and analyst workload. This distinction is important because it shows that the project matured from exploratory modelling into a more deployment-oriented decision framework.

At the API level, Week 21 confirmed that the frozen serving layer behaved correctly for valid requests and for most malformed inputs. Missing fields, extra fields, negative values, and invalid numeric strings were all rejected deterministically. One remaining hardening issue was identified for non-finite numeric inputs (`NaN` and `Infinity`), which surfaced as `500 Internal Server Error` rather than clean validation responses. This issue was documented transparently as a serving-layer improvement point rather than a modelling weakness.

### Month 6 synthesis

Taken together, Weeks 17 to 21 represent the final engineering and deployment maturity phase of the project. During this period, the system evolved from a selected machine learning model into a frozen, testable, portable, cloud-deployed fraud detection service with explicit serving metadata, validation rules, threshold governance, and documented operational behaviour.

This month is therefore the point at which the project becomes strongest from a portfolio and viva perspective. It demonstrates not only that a high-performing fraud model was developed, but also that the model was packaged responsibly, deployed realistically, validated under locked conditions, stress-tested at the API layer, and interpreted through a business-aware serving lens.

### Final Month 6 conclusion

Month 6 concludes the project by showing that the final fraud detection system is reproducible, technically validated, deployment-oriented, and presentation-ready. The model is frozen, the serving contract is explicit, the runtime is containerized, the API is live in the cloud, the threshold policy is operationally justified, and the final evidence has been organized for both report writing and demonstration. In this sense, the closing phase of the project does not simply finalize the implementation; it demonstrates end-to-end ownership of the system from model selection to deployed service delivery.

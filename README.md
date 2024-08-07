# mlops_assignment1_group18
This repository demonstrates the mlops workflow for a simple model served through REST endpoints

## Set Up a CI/CD Pipeline

 Tasks:
     Use a CI/CD tool like GitHub Actions or GitLab CI to set up a pipeline for a sample machine learning project.
     Include stages for linting, testing, and deploying a simple machine learning model.

 * Linting Stage: This stage runs a linting tool (e.g., Flake8) to check the code for style and syntax issues. The configuration includes setting up the environment, installing dependencies, and running the linter.
 * Testing Stage: This stage runs unit tests using a testing framework (e.g., Pytest) to ensure the code functions correctly. The configuration includes setting up the environment, installing dependencies, and running the tests.
 * Deploy Stage: This stage simulates or performs the deployment of a machine learning model. In the dummy deploy step, it can simply echo a message indicating a successful deployment. For actual deployment, it could include steps to deploy to a temporary server or service.
# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in the response:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [ ] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
Answer:
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, ... *
>
Answer:

s201920, s214458, s221336, s250702 , s194504

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
Answer:

We didn't end up using any third-party frameworks, apart from `kagglehub`, this was used in `data.py` to automatically download our data rather than manually downloading it, which contributes to reproducibility.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
Answer:

We used `uv` for managing our dependencies. All dependencies are specified in `pyproject.toml`, separated into main dependencies and optional development dependencies, that is not needed to run the model in production.

We have generated the `requirements.txt` file was using `uv pip compile pyproject.toml` as this was part of a task in the list.

To get an exact copy of our environment, a new team member would need to:

1. Install `uv`
2. Clone the repository
3. Run `uv sync` to install all dependencies and create a locked virtual environment from our pyproject.toml file while updating `uv.lock`. The lock file guarantees that the same package versions are used everywhere, preventing "works on my machine" issues.
4. Now just use `uv run <command>` to execute things in the environment.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
Answer:

From the cookiecutter template we have kept and filled out the core layout: `data/`,`dockerfiles/`, `reports/` `models/`, `notebooks/`, `reports/`, `src/`, `tests/`.

We are ignoring `data/`,`models/` and `wandb/` to keep large or generated artifacts out of Git. We instead track data/model versions via DVC or W&B instead of source control. The `wandb/` directory contains a folder with reports for each run, and is ignored as the records live in the W&B cloud project.

We removed `notebooks/` directory, as we did not use it and `docs/` for the MkDocs site as we did not implement the extra S10 exercises to our project.

Deviations from the vanilla template: added `configs/` for sweep settings for W&B sweeps.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
Answer:

We used Ruff for both linting and formatting our code. Ruff is a fast Python linter that enforces PEP8 standards and automatically formats code to maintain consistency. For type checking, we implemented mypy to validate type hints throughout our codebase. Documentation was handled through Google-style docstrings for all functions and classes. These tools are integrated into our CI/CD pipeline through GitHub Actions and pre-commit hooks, ensuring code quality checks run automatically before code is merged.

These concepts are critical in larger projects for several reasons. Code formatting ensures consistency across the codebase, making it easier for team members to read and understand each other's code. Linting catches potential bugs and code smells early in development. Type hints with mypy provide compile-time error detection, reducing runtime errors and making the code self-documenting. Good documentation through docstrings helps new team members quickly understand function purposes and expected inputs/outputs. Together, these practices reduce technical debt, improve collaboration, and make the codebase more maintainable as it scales.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
Answer:

We implemented 32 tests across multiple test suites. Our unit tests cover data loading and preprocessing functionality (test_data.py), model architecture and forward pass behavior (test_model.py), and API endpoints including root, health, and prediction endpoints (test_api.py). We also have integration tests that verify end-to-end API functionality with a running server, performance tests that measure model inference speed against defined thresholds, and load tests using Locust to validate API performance under concurrent requests. These tests ensure our data pipeline, model, and API work correctly both individually and together.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
Answer:

Our current test coverage is 23% across the codebase. Major gaps are in API request/response paths, dataset preprocessing branches, drift monitoring code, and training/evaluation scripts, which currently sit at low or 0% coverage. Even with 100% coverage we would not assume the code is bug-free: coverage only shows lines executed, not that outputs or edge cases are correct. If a function or class has 100% test coverage it just means that the edge cases that are handled, have tests. It does not mean that all edge cases are handled. The easiest way to achieve 100% test coverage is to handle any edge cases at all (or not even write any code at all). High-value tests must assert behavior (outputs, error handling, performance thresholds) and include integration paths.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
Answer:

Yes, we made extensive use of branches and pull requests in our workflow. Each team member created a new branch when implementing a new feature or fixing a bug. Once coding was complete, the member would pull the latest changes from main and merge them into their branch to ensure compatibility. After pushing their changes, they would create a pull request to merge into main. GitHub Actions automatically ran our test suite on each pull request, and merging was only allowed if all tests passed successfully.

For smaller changes, team members could skip pulling main first and create the pull request directly. GitHub would indicate whether the merge could happen automatically. If merge conflicts arose, the member would resolve them before proceeding. This workflow ensured code quality through automated testing, prevented breaking changes from reaching main, and provided visibility into what each team member was working on through the pull request interface.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
Answer:

Yes. We used DVC to keep all team members on the exact same dataset. Any data curation steps (e.g., removing bad samples) were versioned so cloud training always pulled the latest validated data. DVC stored large files in remote storage while Git tracked only small metadata, so we could `dvc pull`/`dvc push` without bloating the repo.

In CI and unit tests we ran `dvc pull`, allowing tests to use the real dataset automatically instead of relying on mock data. This avoided passing data files around manually, reduced “works on my machine” drift, and kept a clear lineage of which data version was used for each run. Overall, DVC gave us reproducible data access locally, in CI, and when training in the cloud.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
Answer:

We split CI by job so it is clear and fast, and we run it on every push/PR to main. Linting lives in [./github/workflows/linting.yaml](.github/workflows/linting.yaml): `uv sync` (with cache) installs deps, then ruff lint/format checks and mypy run. Unit tests are in [./github/workflows/unit_tests.yaml](.github/workflows/unit_tests.yaml) with a matrix over Ubuntu/Windows/macOS and Python 3.12/3.13; after GCP auth we `dvc pull` to get the tracked dataset, then run pytest with coverage. Integration tests in [./github/workflows/integration_tests.yaml](.github/workflows/integration_tests.yaml) do the same to exercise the API end-to-end across OS/Python. For staged models we have a repository_dispatch workflow that pulls a W&B artifact and runs performance tests: [stage_model.yaml](https://github.com/noanorgaard/MLOps_project/actions/workflows/stage_model.yaml). An example of a triggered workflow can be seen [here](https://github.com/noanorgaard/MLOps_project/actions/runs/21066112061).

We cache installs via setup-uv to keep runs quick. The OS/Python matrix finds platform-specific issues early. We block merges unless ruff, mypy, and tests all pass. Pulling data with DVC inside CI means tests and training always use the same versioned dataset, both locally and in the cloud. Secrets stay in GitHub Secrets (GCP key, W&B key), so workflows can authenticate without hard-coding credentials. Coverage reports run on every test workflow, so we see regressions immediately. By keeping lint, unit, integration, and staged-model checks separate, failures are easy to diagnose and rerun. The repository_dispatch hook lets us test a specific staged model on demand without re-running the whole matrix, which keeps performance checks cheap and targeted.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
Answer:

We used `invoke` task runner with `uv` to standardize how to run experiments. To run locally:
> `uv run invoke train` # runs with defaults (lr=1e-4, batch_size=64, epochs=2)

We use `wandb` as the central hub for hyperparameter management. The `train()` function accepts an optional config dict, which gets passed to `wandb.init()`, and then the hyperparameters are read from `wandb.config`.

We have also created a `sweep_config.yaml` that defines the sweep space and lets W&B automaticallysample hyperparameter combinations to maximize training accuracy. you can launch a sweep using `wandb sweep configs/sweep_config.yaml`


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
Answer:

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
Answer:
We use Weights & Biases (W&B) to keep track of our experiments and compare different training runs within the same project. As shown in the first screenshot, we log the training loss over steps and epochs for multiple runs, including hyperparameter sweeps. Loss is an important metric because it shows how well the model is learning and how quickly it converges for different configurations.

Besides loss, we also track the number of epochs and training steps for each run. This makes it easier to compare experiments fairly, especially in sweeps where some runs may train longer than others. Looking at loss together with epochs helps us understand whether better results come from better learning or simply from longer training, and it can also hint at overfitting.

We also log training accuracy and validation accuracy. Training accuracy shows how well the model fits the training data, while validation accuracy tells us how well it generalizes to unseen data. Comparing these two metrics helps us spot overfitting or underfitting and decide if we need changes like early stopping or different hyperparameters.

Finally, we log image artifacts during training, as seen in the media panel. These images let us visually check what the model is producing and can reveal issues that are not always obvious from just numbers.

Overall, combining these metrics with logged images gives us a good overview of how the model behaves and helps us compare experiments in a structured way.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
Answer:

We used virtual machines for training of the model, and uploaded the resulting model to WandB.

We used the artifact registry to store our docker images, that could then be run with Cloud run. This combination we used to deploy our inference api, and the resulting prediction was stored in a Bucket. We also used this combo for deploying our drift api, that read the predictions(created through the inference API) from the bucket. 

Buckets was also used to store training data. 

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
Answer:

We used VM's for model training by running our training code inside a container as a managed custom
training job.

we used the VM type `n1-standard-4` (4 vCPUs and ~15 GB RAM). We did not manage to get GPU access, but luckily were a patient bunch. The model was pretty ass as a result of this
The vm was used image classifier training runs and for logging artifacts/metrics to Weights & Biases. The container image
was built in Cloud Build, stored in Artifact Registry, and then referenced from the job spec (see `config_cpu.yaml`).

For deployment of our inference and drift API's we used Cloud Run, that does not specify the VM type as it runs it automatically by itself. 

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
Answer:

![Buckets overview](figures/Screenshot%202026-01-21%20at%2010.18.47.png)

![Buckets overview](figures/Screenshot%202026-01-21%20at%2010.23.24.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
Answer:

![alt text](figures/image-3.png)-

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
Answer:
![alt text](figures/image-1.png)
![alt text](figures/image-2.png)

A common(fælles) project was used aswell personals (therefore the two images) 


### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
Answer:

We trained the model in cloud using the engine. The training was done through the linux based Compute engine vitual machine. Here we ran our training script and made the data availeble by mounting the data it from the cloud storage(bucket) on which allowed us to train without altering the training code.


## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
Answer:

We built a `FastAPI` service in `api.py`. By default it pulls the W&B artifact (using `WANDB_API_KEY`) alias ai_vs_human_model:latest, so it loads whatever run we last aliased as latest. If we set WANDB_SWEEP_ID, the API will instead fetch the best model from that sweep using the configured metric (train/epoch_acc by default). The API auto-selects GPU/MPS/CPU, and creates a `/predict` endpoint for image uploads on a localhost. Incoming images are validated, resized to 224×224, passed through our classifier, and the response returns class (AI or human) plus the confidence.
The app runs locally with `uv run uvicorn ai_vs_human.api:app --reload` or via the inference Dockerfile in `dockerfiles/api.dockerfile`. For offline use, the service can be pointed to a local artifact directory instead of pulling from W&B.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
Answer:

We containerized the FastAPI service with api.dockerfile and run it locally via `docker run -p 8000:8000 api:latest` after building. In GCP we used Cloud Build + Artifact Registry to build/push the image, then deployed to Cloud Run with the `WANDB_API_KEY` and artifact parameters as env vars. Invocation: `curl -X POST -F "file=@sample.jpg" https://ai-vs-human-api-onyhgpfxqq-ew.a.run.app/predict` returns the class and confidence. For local dev, `uv run uvicorn ai_vs_human.api:app --host 0.0.0.0 --port 8000 --reload` works with a `.env` containing `WANDB_API_KEY`.



### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
Answer:

For unit testing we used FastAPI's TestClient with pytest in `test_api.py`. Tests cover the root, health, and prediction endpoints, error handling for edge cases (invalid files, model failures, corrupted images), response validation (status codes, JSON schema, prediction values), and image preprocessing. We used mocked models to test the API logic without requiring a trained model. This returns predictable tensor outputs, allowing us to test the API logic (preprocessing, response formatting, error handling) without loading an actual trained model from W&B, to make our tests faster and independent of external dependencies.

For load testing we used `Locust` (tests/loadtests/locustfile.py) and ran headless via `uv run invoke load_test_locust --host http://localhost:8000 --users 50 --rate 5 --time 2m`, which produced `reports/load_test_report.html`. Under that load the service stayed stable with no errors and p95 latency around 300–400 ms locally.

The app handled the load without crashing, we didn't test higher loads to find the breaking point.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
Answer:

We added `Prometheus` metrics to the API, able to be accessed exposed at a `/metrics` endpoint, tracking request counts, latency, prediction counts, confidence histograms.

Locally we can scrape this with Prometheus or inspect via `curl`. In production, Cloud Monitoring (managed Prometheus) would scrape the `/metrics` endpoint in Cloud Run to set up alerts on elevated latency/error rates and track drift via confidence distributions.

We haven't wired a full external monitoring service yet, but the metrics are instrumented and ready for scraping and alerting to catch performance degradation or model drift over time. It would be smart to add Cloud run drift detection and automated alerts to compare current prediction confidence distributions to baselines to catch model degradation early. And get notified when latencu spikes or error rates increase. Without it, we are flying blind and will not know that the API degraded until users report problems.


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
Answer:

We used 323 kr of credits, all of it on Cloud Storage (bucket). The bulk of the spend came from CI runs because our unit tests pulled real data from the bucket via DVC so tests could run directly on the actual dataset. This ensured reproducibility across teammates and CI but increased bucket egress and access during frequent PRs.

Working in the cloud was practical for building and running containers and for storing/serving data from a single place. The main friction was getting everyone authenticated to the same project and handling billing setup, which added some overhead before we could all run workflows consistently.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
Answer:

Yes, we implemented energy consumption monitoring using Zeus during model training. Zeus is a framework that tracks GPU energy usage in real-time, providing insights into the environmental and computational cost of training runs. We integrated Zeus into our training pipeline to measure the energy consumed by each training session. This gave us visibility into how much power our experiments used, which is important for understanding the carbon footprint and operational costs of model training, especially when scaling up or running hyperparameter sweeps. By monitoring energy consumption, we could identify more efficient training configurations and make informed decisions about trade-offs between model performance and resource usage.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
Answer:

The figure shows the overall architecture of our system and how the different tools are connected.

The workflow starts on the developer’s local machine, where we work with the code and the data. The data is tracked using DVC, which helps with versioning datasets and keeping experiments reproducible. From the local setup, we run the training pipeline, which includes training, evaluation, and visualization scripts.

During training, we use Weights & Biases (W&B) to log metrics such as loss and accuracy, along with model artifacts and images. We also run hyperparameter sweeps in W&B to test different configurations and compare results. The trained models are stored as artifacts, and the best or most recent model can later be selected for deployment.

When code is pushed to the GitHub repository, it automatically triggers GitHub Actions. The CI pipeline runs tests and checks the code, and if everything passes, a Docker image is built. This image is pushed to a container registry and used to deploy the model as an API service.

Users interact with the system through the API, which loads the selected trained model and returns predictions. Overall, this setup connects development, experimentation, testing, and deployment in a structured and automated way.

The final part of interaction between the user and docker/API is not fully realized, and requires some manual input for certain aspects. Additionally, the API key for wandb is not integrated into the system completely, and needs a local environment variable to be set to acces certain artifacts.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
Answer:

Student s201920 was in charge of developing of setting up the initial cookie cutter project and setting up and configuring the API and configuring the connection to wandb.
Student s214458, was in charge of implementing wandb logging and sweeping and major changes to train.py.
Student s221336 was in charge of setting up DVC with GCP for the project. They was also responsible for configuring the contionous integration setup to run tests on pull requests and pushes to main. Finally they implemented the tests on staged models workflow and integrated the Zeus framework for monitoring energy consumption during training.
Student s250702,
Student s194504, was in charge of training our models in the cloud and deploying them afterwards.

All members contributed to code by sharing the same Git.

We have used ChatGPT to help debug our code and set up docker files. Additionally, we used GitHub Copilot to help write some of our code and ask about "how to Git", and we have set up an `AGENT.md` file to streamline our prompt answers.

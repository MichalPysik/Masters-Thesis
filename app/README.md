# System for searching and analyzing videos of traffic

The system consists of a web application, installed and managed using Poetry (which automates the creation and management of virtual environments and simplifies dependency management), along with additional services running in containers managed by Docker Compose. The backend, which also integrates the AI models, is written in Python and uses FastAPI. The frontend, written using Vue.js,is in the form of a single-page application (SPA), and is served directly by the backend as a ”sub-application” using FastAPI’sStaticFiles.

## Quickstart guide

1. Make sure Poetry, Docker Compose, NPM, CUDA, and Rust compiler are installed on your (Linux) system
2. Configure the web application by modifying the `.env` file (create it by copying `.env.example` if it doesn't exist)
3. Install frontend's dependencies and build the frontend (you can skip this step if you set `USE_FRONTEND=False` in `.env`): `cd frontend && npm install && npm run build`
4. Install the application and its dependencies in a virtual environment: `poetry install`
5. Setup and run the additional services running inside containers (Milvus DB, Minio): `docker compose up`
6. Run the web application: `poetry run app`

## Folder structure

* **/src** - The Python source code (backend)
* **/frontend** - The SPA frontend implemented in vue.js, which is mounted to the backend via FastAPI's StaticFiles
* **/tmp** - A folder used by the web application to use temporary files for MLLM analysis with certain models
* **/volumes** - Data used by the services running inside containers (Milvus DB, Minio)
* **.env** - The defined environment variables (configuration of the application)
* **.env.example** - An example of how a `.env` file should look
* **docker-compose.yml** - Defines and runs the additional services containers (Milvus DB, Minio)
* **pyproject.toml** - Configures the Python project metadata and dependencies using Poetry
* **poetry.lock** - Records the exact versions of all Python packages used in the project to ensure consistency (do not manually modify this file)

## Usage

This section describes how to use the application using the web-based frontend. If you wish to call the backend endpoints yourself, refer to the REST API Swagger UI documentation available at `http://host:port/docs` or the endpoints description found in the thesis' text (chapter 5, section 3). In the web UI, the menu for switching between the three separate views described below is always present in the upper-left corner.

### Video upload and data management

* The list shown in the middle of the screen shows all data present in the system (videos in Minio bucket, embeddings in Milvus DB collection). Each entry consists of the video name, the duration (sampled on-demand from the Minio bucket), flag marking whether the video file is present in the Minio bucket, the number of embedding entries found in Milvus DB, and buttons for performing actions related to the video (delete, synchronize).
* To upload a new video (top of the screen), choose the video file, the sampling FPS, flag whether to only upload the video to the bucket (skips the embedding extraction for later), and press the upload button.
* Synchronizing a single video deletes and recreates all related embeddings stored in the Milvus DB collection (if the video file is present in the Minio bucket, otherwise it just deletes all related data just as the delete button does). Warning: The sampling FPS value selected at the top (upload) affects the embedding recreation!
* Synchronizing all videos:
  - Without force bucket mirroring - Deletes all embedding entries in the Milvus DB collection without a corresponding video file present in the Minio bucket, creates embeddings for all videos with exactly 0 existing embedding entries.
  - With force bucket mirroring - Deletes all embedding entries in the Milvus DB collection, (re)creates enbeddings for all videos present in the Minio bucket while first deleting the entire collection and creating a new one (this is especially useful when switching between different multimodal embedding models).

### Embedding search

* Enter the search query, select the number of results to retrieve, and press the search button.
* To analyze a video segment centered around a certain search result, click the 'Analyze this segment' button to be automatically redirected.

### Video MLLM analysis

1. Configure the analysis session at the top (unless redirected from the search view) - this includes the video file name, the start and end timestamps specifying the video segment to analyze, and the number of frames that will be sampled and inputted to the MLLM.
2. Press the 'New analysis button' - this opens a video player and a chat window (with "frozen" parameters unaffected by changing the ones at the top).
3. Chat with the currently configured MLLM about the video segment.

## Switching and configuring the available AI models

* To switch the multimodal embedding model, change the corresponding variable in `.env` and restart the application. You will also have to sychronize all videos with force bucket mirroring set to True to recreate the entire Milvus DB collection (with the correct embedding vector size). If you want to use the BLIP model, you will have to comment/uncomment certains lines inside `pyproject.toml` (the comments there will guide you) and type `poetry install` again.
* To switch the MLLM, change the corresponding variable in `.env` and restart the application. You might have to modify the `transformers` version inside `pyproject.toml` for LLaVA-OneVision to work.
* Configurations of the locally run models can be changed inside `src/__init__.py`.
* To use GPT-4o (remotely only), you will have to configure certain `.env` variables depending on whether you are using standard OpenAI or Azure service:
  - OpenAI: `OPENAI_USE_AZURE=False, OPENAI_API_KEY` and optionally, if you want to use a custom endpoint, configure the `OPENAI_CUSTOM_ENDPOINT` variable
  - Azure: `OPENAI_USE_AZURE=True, OPENAI_API_KEY={azure_api_key}, OPENAI_CUSTOM_ENDPOINT={azure_endpoint}, OPENAI_AZURE_DEPLOYMENT_NAME, OPENAI_AZURE_API_VERSION`

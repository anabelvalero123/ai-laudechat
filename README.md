# Laude Chat 

Laude Chat is a Python-based application designed to facilitate dialogues about document contents, enabling users to efficiently retrieve specific information. This tool supports intuitive, natural language queries, utilizing the principles of Retrieval-Augmented Generation (RAG) to accurately generate responses that are closely aligned with the information contained within the documents.

![Laude Chat App Screen](/images/LaudeChatScreen.png "Laude Chat App")

## Main Features

* Unimodal Retrieval Augmented Generation (RAG)
* Text embeddings: textembedding-gecko@003
* LLM model: PaLM2 (text-bison@002)
* Vectorstore: FAISS


## Code Structure

        LaudeChat/
        │
        ├── app/                            # Main directory of the application
        │   ├── images/                     # Directory for the application images           
        │   │   ├── laude_logo.jpg        
        │   │   └── ... 
        │   │
        │   ├── templates/                  # HTML templates for the application
        │   │   └── htmlTemplates.py        # Manages HTML templates
        │   │
        │   ├── pages/                      # Directory for Streamlit application pages
        │   │   ├── CHANGELOG.md            # Record of changes and updates to the application
        │   │   ├── Retrieval_Source.py     # Page for displaying retrieval sources
        │   │   └── Version_Updates.py      # Page displaying application updates
        │   │
        │   ├── LaudeChat.py                # Main script of the application
        │   ├── variables.py                # Application configuration variables
        │   └── vectorstore_utils.py        # Functions related to the vector store
        │
        ├── Dockerfile                      # Defines the environment for containerizing the application
        ├── requirements.txt                # Project dependencies
        ├── README.md                       # Project documentation
        └── images/                         # Directory for documentation images          
            ├── LaudeChatScreen.png       
            └── ... 

## Build and Deploy the Application to Cloud Run

Ensure that you have cloned this repository and is your active working directory for the rest of the commands.

To deploy the Streamlit Application in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), we need to perform the following steps:

1. Your Cloud Function requires access to two environment variables:

   - `GCP_PROJECT` : This the Google Cloud Project Id.
   - `GCP_REGION` : This is the region in which you are deploying your Cloud Function. For e.g. us-central1.
  
    These variables are needed since the Vertex AI initialization needs the Google Cloud Project Id and the region. 

    In Cloud Shell, execute the following commands:

    ```bash
    export GCP_PROJECT='<Your GCP Project Id>'  # Change this
    export GCP_REGION='europe-west1'            # Make sure region is supported by Model Garden. If it's not suported try 'us-central1'.
    ```

2. We are now going to build the Docker image for the application and push it to Artifact Registry. To do this, we will need one environment variable set that will point to the Artifact Registry name. We have a command that will create this repository for you.

   In Cloud Shell, execute the following commands:

   ```bash
   export AR_REPO='<REPLACE_WITH_YOUR_AR_REPO_NAME>'  # Change this
   export SERVICE_NAME='laude-chat-app' # This is the name of the Application and Cloud Run service. Change it if you'd like. 
   gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
   gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
   gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
   ```

3. The final step is to deploy the service in Cloud Run with the image that we built and pushed to the Artifact Registry in the previous step:

    In Cloud Shell, execute the following command:

    ```bash
    gcloud run deploy "$SERVICE_NAME" \
      --port=8080 \
      --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
      --allow-unauthenticated \
      --region=$GCP_REGION \
      --platform=managed  \
      --project=$GCP_PROJECT \
      --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION
    ```

On successfully deployment, you will be provided a URL to the Cloud Run service. You can visit that in the browser to view the application that you just deployed. Type in your queries and the application will prompt the Vertex AI Text model and display the response.

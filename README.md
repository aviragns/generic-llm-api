# generic-llm-api
generic-llm-api


Build docker image

docker build . -t flaskllm:one

Run docker (change credentials path according to your local system)

docker run -v "$HOME/.config/gcloud/application_default_credentials.json":/gcp/creds.json:ro --env GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json --env GCP_PROJECT_ID=hsbc-wsit-gen-ai -p 5000:5000 flaskllm:one
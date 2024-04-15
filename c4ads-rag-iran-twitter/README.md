# C4ADS RAG

# Docs
- [LLamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)

## Dependencies

### Data
Sample data can be downloaded at this link https://www.wikiran.org/leaks/sahara-thunder


### Docker Compose
This app uses Docker Compose to run all the necessary services in a containerized environment.

To run this service, you simply need to build and start up the relevant services:
```
docker-compose build && docker-compose up
```

### Redis Cache

This application relies on a local `redis` cache running at host="127.0.0.1" & port=6379

### Open AI Key

The user of this script must load a local OPEN_AI_KEY at a .env file
```
cp .sample_env .env
# add the relevant fields found within the sample .env file
```
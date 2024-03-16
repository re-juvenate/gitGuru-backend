Git Guru
=====
# Backend
-----

## AI

### Quickstart

AI Model Dependencies:
(download the following from ollama)
```sh
ollama pull dolphin-phi:2.7b-v2.6-q8_0

ollama pull nomic-embed-text:latest	

```

Optionally, run `ollama pull mithun/magicoder:6.7B-S-DS-Q4_K_M` for ussing local models entirely


Edit `settings.yaml` to include the model paths and configurations.

Install SearXNG and start it 
```sh
docker pull searxng/searxng
export PORT=8080
docker run --rm \
             -d -p ${PORT}:8080 \
             -v "${PWD}/searxng:/etc/searxng" \
             -e "BASE_URL=http://localhost:$PORT/" \
             -e "INSTANCE_NAME=my-instance" \
             searxng/searxng
```

For stopping docker  ` docker container stop {container_id} `


-----
By, team rejuvenate.
-----


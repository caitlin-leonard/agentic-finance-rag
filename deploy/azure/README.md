# Deploying to Azure Container Apps

The image is provider-agnostic; on Azure it runs against **Azure OpenAI**
(the same stack Cognizant's ACE program references as Azure AI Foundry) and
**Azure Database for PostgreSQL Flexible Server** with the `pgvector` extension.

```bash
# 1. Build & push
az acr build -r <registry> -t agentic-finance-rag:latest .

# 2. Postgres + pgvector
az postgres flexible-server create -n finrag-pg -g <rg> --version 16
# then: CREATE EXTENSION vector;  (see src/finrag/retrieval/store.py schema)

# 3. Container App
az containerapp create \
  -n finrag-api -g <rg> --environment <env> \
  --image <registry>.azurecr.io/agentic-finance-rag:latest \
  --target-port 8000 --ingress external \
  --secrets azure-key=<key> db-url=<conn> \
  --env-vars FINRAG_PROVIDER=azure FINRAG_USE_PGVECTOR=true \
             FINRAG_AZURE_OPENAI_ENDPOINT=<endpoint> \
             FINRAG_AZURE_OPENAI_API_KEY=secretref:azure-key \
             FINRAG_DATABASE_URL=secretref:db-url
```

Scale-to-zero + HTTP autoscaling are handled by Container Apps; the `/health`
endpoint backs the readiness probe.

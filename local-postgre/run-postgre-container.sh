docker run -d \
  --name postgres-db \
  --restart unless-stopped \
  -p 32345:5432 \
  postgres-db-pgvector
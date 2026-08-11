FROM node:20-alpine AS builder
WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Extract pre-installed node_modules (avoids npm install in container)
ADD node_modules.tar.gz ./

# Run npm install to ensure integrity (uses local node_modules as cache)
RUN npm install --legacy-peer-deps

# Copy source code
COPY frontend/. .

# Build
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

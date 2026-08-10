# ============================================================================
# React Frontend Dockerfile
# Multi-stage build: Node.js builder → nginx serving
# ============================================================================

# Stage 1: Build the React app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better caching
COPY frontend/package*.json ./

RUN npm install --legacy-peer-deps

# Copy source code and build
COPY frontend/ ./

ARG API_URL=http://localhost:5000
ENV VITE_API_URL=${API_URL}

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Use existing nginx config from docker/nginx/
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
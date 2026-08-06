# =============================================================================
# FRONTEND DOCKERFILE - React 18 + TypeScript
# =============================================================================

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

RUN npm ci --only=production

# Copy source code
COPY frontend/src ./src
COPY frontend/public ./public
COPY frontend/vite.config.ts ./
COPY frontend/tsconfig.json ./

# Build for production
RUN npm run build

# Stage 2: Serve with Nginx (lightweight)
FROM nginx:alpine AS production

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Security headers
COPY nginx/security.headers /etc/nginx/security.headers

# Run as non-root
USER nginx

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

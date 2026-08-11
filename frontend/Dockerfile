FROM node:20-alpine AS builder
WORKDIR /app

# ARG proxy support
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}

# Copy from frontend/ subdirectory (context is parent ATA/)
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps --verbose
COPY frontend/. .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# Point to nginx.conf in docker/nginx/ folder
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
FROM node:20-alpine AS builder
WORKDIR /app

# Corporate proxy support (adjust for charite network)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}

COPY package*.json ./
RUN npm install --legacy-peer-deps --verbose

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
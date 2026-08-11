FROM nginx:alpine

# Copy pre-built frontend dist (already built locally on VM)
COPY frontend/dist /usr/share/nginx/html

# Use nginx config for SPA routing
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

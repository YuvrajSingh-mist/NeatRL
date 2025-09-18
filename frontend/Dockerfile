FROM ruby:3.2-alpine

# Install dependencies
RUN apk add --no-cache \
    build-base \
    linux-headers \
    git \
    nodejs \
    npm \
    curl

# Set working directory
WORKDIR /app

# Copy Gemfile and Gemfile.lock
COPY Gemfile* ./

# Install bundler and handle lockfile compatibility
RUN gem install bundler && \
    bundle config set --local path 'vendor/bundle' && \
    bundle config set --local deployment false && \
    bundle install --jobs 4 --retry 3

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 4000

# Start Jekyll with development server
ENV JEKYLL_ENV=development
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--port", "4000", "--force_polling", "--watch", "--config", "_config.yml,_config_dev.yml"]

#FROM node:10.13.0-alpine
FROM node:14.4.0-alpine
# Env
ENV ENV_NAME development
ENV EGG_SERVER_ENV development
ENV NODE_ENV development
ENV NODE_CONFIG_ENV development


# Create Directory for the Container
WORKDIR app
# Only copy the package.json file to work directory

#COPY package.json .
COPY . /app
# Install all Packages
# RUN npm run compile
#RUN npm install --unsafe-perm

RUN apk --no-cache --virtual build-dependencies add \
    python \
    make \
    g++ \
    && npm install --unsafe-perm\
    && apk del build-dependencies



# Copy all other source code to work directory
ADD . /app

# TypeScript
#RUN npm run tsc
# Start

CMD [ "npm", "start" ]
EXPOSE 8081

FROM public.ecr.aws/docker/library/node:20.12.0-alpine

WORKDIR /app

COPY package*.json ./

RUN npm i

COPY . .

EXPOSE 9000

RUN npm run build

CMD ["npm", "start"]

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget

# install eosio tools
RUN wget https://github.com/AntelopeIO/leap/releases/download/v4.0.1/leap_4.0.1-ubuntu22.04_amd64.deb

RUN apt-get install -y ./leap_4.0.1-ubuntu22.04_amd64.deb

RUN mkdir -p /root/nodeos
WORKDIR /root/nodeos
COPY config.ini config.ini
COPY contracts contracts
COPY genesis genesis

EXPOSE 42000
EXPOSE 29876
EXPOSE 39999

CMD sleep 9999999999

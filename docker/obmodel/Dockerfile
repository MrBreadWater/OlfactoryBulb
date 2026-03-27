FROM neuron:arm64

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y git x11vnc xvfb openbox menu
EXPOSE 5920
ENV DISPLAY=:20

RUN echo 'openbox-session 2> /dev/null || true &' >> /root/.bashrc

COPY . /OlfactoryBulb
WORKDIR "/OlfactoryBulb"

RUN pip3 install -r requirements-arm64.txt

WORKDIR "/OlfactoryBulb/prev_ob_models"
RUN ./compile_mod.sh
WORKDIR "/OlfactoryBulb"

RUN echo 'cd /OlfactoryBulb' >> /root/.bashrc
EXPOSE 8888

CMD ["sh", "-c", "\
     x11vnc -nopw -create -rfbport 5920 -forever & disown;\
     jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser;\
"]

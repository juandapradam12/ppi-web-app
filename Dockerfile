FROM python:3.8.16
ENV PYTHONUNBUFFERED=1

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . .

ADD player_performance_index.py /
ADD ppi_app.py /

CMD [ "python", "./player_performance_index.py" ]
CMD [ "python", "./ppi_app.py" ]


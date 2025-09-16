#!/bin/bash

python run.py -i 1000_bad_weather_with_llm.json -o factowl_bad_weather.json
python run.py -i 1000_cars_with_llm.json -o factowl_cars.json
python run.py -i 1000_rivers_with_llm.json -o factowl_rivers.json

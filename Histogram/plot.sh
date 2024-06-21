#!/bin/bash

cmake -B build && make -C build ;
./hist.out > data.csv ;
python3 plot.py ;

rm -f data.csv ;

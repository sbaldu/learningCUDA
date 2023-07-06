#!/bin/bash

make ;
./hist.out > data.csv ;
python3 plot.py ;

rm -f data.csv ;

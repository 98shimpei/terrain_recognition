#!/bin/bash
./rm_generate_terrains.sh
./generate_terrain.py 2000
sleep 1
./real_steppable_region_learning.py 200000 10 f
./rm_generate_terrains.sh
./generate_terrain.py 2000
sleep 1
./real_steppable_region_learning.py 10000 10 wv

sleep 1
./real_steppable_region_learning.py 200000 10 fw
./rm_generate_terrains.sh
./generate_terrain.py 2000
sleep 1
./real_steppable_region_learning.py 10000 10 wv

sleep 1
./real_steppable_region_learning.py 200000 10 fw
./rm_generate_terrains.sh
./generate_terrain.py 2000
sleep 1
./real_steppable_region_learning.py 10000 10 wv

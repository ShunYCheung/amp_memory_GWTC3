#!/bin/bash

for ((i=1;i<=100;i++)); 
do 
   bilby_pipe_generation inject_GW170818_mem_A1_run$i.ini
   echo $i
done


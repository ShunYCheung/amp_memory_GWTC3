#!/bin/bash

for ((i=0;i<=19;i++)); 
do 
   bilby_pipe inject_GW150914_A1_run$i.ini
   echo $i
done


#!/bin/bash

echo ${@}

dir="data/"

file=$1
flag=$2

if [[ ! "$flag" == "" ]]
then
	flag="-$flag"
fi

if [[ "$file$" =~ .*"_SENS".* ]]
then
	type="s"
else
	type="m"
fi


echo "python toolplot.py -s $type -f $dir$file.pkl $flag"
python toolplot.py -s $type -f $dir$file.pkl $flag

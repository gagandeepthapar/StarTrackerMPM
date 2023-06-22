#!/bin/bash

# Define the directory
dir_path="./data"
pkl=".pkl"
json=".json"
dest="/Users/gagandeepthapar/Desktop/School/AERO/MS/Thesis/Documents/StarTrackerThesis/chapters/5_Univariate_Effect_Analysis/images"
# Loop over files in directory

okarr=()

for file in "$dir_path"/*.pkl
do
  # Execute command for each file
	base=$(basename "$file" "$pkl")
	par=$(basename $(dirname "$file"))
	pklfile="$par/$base$pkl"

	echo $pklfile



#	for imgfile in "$dest"/
#	do
#		fname=$(basename "$imgfile")
#		if [[ "$base.png" == $fname ]]
#		then
#			echo "$base.png exists! Skipping"
#			echo $imgfile
#			FLAG=1
#		fi
#	done

		if [[ ! "${okarr[*]}" =~ "$base" ]]
		then
			if [[ "$base" =~ .*"SENS".* ]]
			then
#				continue
				echo python toolplot.py -f $pklfile -s s
				python toolplot.py -f $pklfile -s s -p
			else
				echo python toolplot.py -f $pklfile -s m
				python toolplot.py -f $pklfile -s m -p
			fi

			echo " "
		fi

done

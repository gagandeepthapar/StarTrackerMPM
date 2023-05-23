#!/bin/bash

# Define the directory
dir_path="./data"
comp="CENTROID"
json=".json"
pkl=".pkl"
sens="_SENS"

PHILATTAG="F_ARR_PHI F_ARR_THETA"
EPSLATTAG="F_ARR_EPS_X F_ARR_EPS_Y"
PHILONGTAG="F_ARR_PSI"
EPSLONGTAG="F_ARR_EPS_Z"
FOCTAG="FOCAL_LENGTH MAX_MAGNITUDE"
STARS="MAX_MAGNITUDE"
SWTAG="BASE_DEV_X BASE_DEV_Y"

namearr=(
"STARS"
"EPS_LAT"
"EPS_LONG"
"EPS"
"PHI_LAT"
"PHI_LONG"
"PHI" 
"FOCLEN"
"CENTROID"
)

tagarr=(
"$STARS"
"$EPSLATTAG"
"$EPSLONGTAG"
"$EPSLATTAG $EPSLONGTAG"
"$PHILATTAG"
"$PHILONGTAG"
"$PHILATTAG $PHILONGTAG"
"$FOCTAG"
"$SWTAG"

)

for i in ${!namearr[@]}; do
	echo "$i: ${namearr[$i]}: ${tagarr[$i]}"
done


# Loop over files in directory
for file in "$dir_path"/*.json
do
  # Execute command for each file
	base=$(basename "$file" "$json")
	par=$(basename $(dirname "$file"))
	jsonfile="$par/$base$json"
	pklfile="$par/$base$pkl"
	sensfile="$par/$base$sens$pkl"

	echo $base

	# IDEAL CASE
	if [ "$base" == "BASE" ]
	then
		if [ ! -e "$pklfile" ]
		then
			echo "START MC: $base"
			echo python driver.py -cam $jsonfile -sw i -n 1000 -name $base
			python driver.py -cam $jsonfile -sw i -n 1000 -name $base
			echo "END MC: $base"
		fi

	# CENTROID CASE
	elif [ "$base" =~ .*"CENTROID".* ]
	then

		if [ ! -e "$pklfile" ]
		then
			echo "START MC: $base"
			echo python driver.py -cam i -sw $jsonfile -n 20000 -name $base 
			python driver.py -cam i -sw $jsonfile -n 20000 -name $base 
			echo "END MC: $base"
		fi

		if [ ! -e "$sensfile" ]
		then
			echo "START SENSITIVITY: $base"
			echo python driver.py -cam i -sw $jsonfile -sim S -name $base$sens -par BASE_DEV_X BASE_DEV_Y
			python driver.py -cam i -sw $jsonfile -sim S -name $base$sens -par BASE_DEV_X BASE_DEV_Y
			echo "END SENSITIVITY: $base"
		fi

	else

		if [ ! -e "$pklfile" ]
		then
			echo "START MC: $base"

			if [[ ! "$base" =~ "FULL".* ]]
			then
				echo python driver.py -cam $jsonfile -sw i -n 20000 -name $base
				python driver.py -cam $jsonfile -sw i -n 20000 -name $base
				
			else
				echo python driver.py -cam $jsonfile -n 20000 -name $base
				python driver.py -cam $jsonfile -n 20000 -name $base
			fi

			echo "END MC: $base"
		fi

		if [ ! -e "$sensfile" ]
		then
			if [[ "$base" =~ "FULL".* ]]
			then
				continue
			fi

			echo "START SENSITVITY: $base"
			for i in ${!namearr[@]}; do
				if [ ${namearr[$i]} == $base ]
				then
					break
				fi
			done
			echo python driver.py -cam $jsonfile -sw i -sim S -name $base$sens -par ${tagarr[$i]} 
			python driver.py -cam $jsonfile -sw i -sim S -name $base$sens -par ${tagarr[$i]} 
			echo "END SENSITIVITY: $base"
		fi
	fi	
done

source plot.sh

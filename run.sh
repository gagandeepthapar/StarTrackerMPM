#!/bin/bash

# Define the directory
dir_path="./data"
comp="CENTROID"
json=".json"
pkl=".pkl"
sens="_SENS"
new="_NEW"

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

	num="20000"
	sw="i"
	cam=$jsonfile

	# base case
	if [[ "$base" == "BASE" ]]
	then
		sw="i"
		cam="i"
		num=1000
	fi

	# check full
	if [[ "$base" =~ .*"FULL".* ]]
	then
		sw="data/CENTROID.json"
	fi

	# check centroid
	if [[ "$base" =~ .*"CENTROID".* ]]
	then
		
		if [[ ! $base =~ .*"FULL".* ]]
		then
			cam="i"
		
		else
			jsonfile="$par/${base#"FULL"}.json"

		fi

		sw=$jsonfile
	fi

	param=""
	if [[ "${namearr[*]}" =~ "$base" ]]
	then
#		echo "EXISTS"

		for i in ${!namearr[@]}; do
			if [ ${namearr[$i]} == $base ]
			then
				break
			fi
		done
		param=${tagarr[$i]}
	
	fi

	if [[ ! $param == "" ]]
	then
		param="-par $param"
	fi
	

	#echo "-n $num -sw $sw -cam $cam $param" 

	echo "MC"
	echo "python driver.py -n $num -sw $sw -cam $cam -name $base"
	python driver.py -n $num -sw $sw -cam $cam -name $base
	echo " "

	if [[ ! "$param" == "" ]]
	then
		echo "SENS"
		echo "python driver.py -sim s -sw $sw -cam $cam $param -name $base$sens"
		python driver.py -sim s -sw $sw -cam $cam $param -name $base$sens
	fi
	echo " "
	
done

#source plot.sh
#

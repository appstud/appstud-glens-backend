
get_root_path() {

	IN=$1
        IFS="/" read -r -a splitted_path <<< "$IN"

        unset splitted_path[-1]	
        unset splitted_path[-1]	
	path=""
	for split in "${splitted_path[@]}"
	    do
		path="${path}/${split}"
	    done
	echo ${path:1} 

}

print_header() {
    echo " launch scripts for age estimation training/evaluation inside docker - GLENS"
    print_delimiter
    echo ""
}

print_delimiter() {
    echo "--------------------------------------------------------------"
}

print_help()
{   
    echo "Usage:"
    echo "  ./docker_run_script.sh run [script_name]"
    echo ""
    echo "actions:"
    echo ""
    echo "help: print this help message"
    echo ""
    echo "list-scripts: list all available scripts to run" 
    echo ""
    echo "run: run the provided script" 
    echo ""
                                    
}


if [ ! -n "$1" ]
then
    print_header && print_help && exit 0
fi


case $1 in
    help)
        shift
        print_help
    ;;
    list-scripts)
        shift
        ls scripts 
    ;;
    run)
        shift
	root_path="$(get_root_path $(pwd))"
	echo ${root_path}
	sudo docker run --gpus all -p 5000:5000 -v ${root_path}:/tmp/src/app  -w /tmp/src/app/beard_detection_training/src mytensorflow bash scripts/$@
    ;;
    *) echo "Unknown action $1"
    ;;
esac












for i in 0 #1 2 3 4 5 6 7 8 9
do
    docker run --name imgen$i -dit -v ~/code/celest/data/image_train_val:/data/image_train_val imgen
    docker update --memory 1g --memory-swap 2g imgen$i
    
    docker cp ssc_gen.yml imgen$i:/
    docker exec -d imgen$i ./screenshot.sh
done

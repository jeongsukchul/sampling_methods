
# for seed in 0 1 2
# do
#     python run.py -m seed=$seed algorithm=smc target=$task 
# done
# for seed in 0 1 2
# do
#     python run.py -m seed=$seed algorithm=fab target=$task 
# done

task="gaussian_mixture40"

for seed in 0 1 2
do
    python run.py -m seed=$seed algorithm=gmmvi target=$task 
done

task="student_t_mixture"

for seed in 0 1 2
do
    python run.py -m seed=$seed algorithm=gmmvi target=$task 
done

task="funnel"

for seed in 0 1 2
do
    python run.py -m seed=$seed algorithm=gmmvi target=$task 
done
# for seed in 0 1 2
# do
#     python run.py -m seed=$seed algorithm=nfvi target=$task
# done

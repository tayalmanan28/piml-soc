#!/bin/bash
# base_name="IP"
# models="exact"
# for model in $models; do
#     for j in $(seq 3 3);
#       do
#           exp_name="${base_name}_${model}_seed_${j}"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "IP_${model}" --wandb_group training --wandb_name ${exp_name} --dynamics_class InvertedPendulum --minWith target --tMax 2 --num_epochs 100000 --counter_end 100000 --num_nl 128 --lr 2e-5 --deepReach_model ${model} --seed $j
#           wait
#           sleep 50
#           # if [ ${model} != "reg" ]; then
#           #   wait
#           #   exp_name="${base_name}_${model}_seed_${j}_nopretrain"
#           #   python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project IP --wandb_group training --wandb_name ${exp_name} --dynamics_class InvertedPendulum --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --deepReach_model ${model} --seed $j
#           #   wait
#           #   sleep 100
#           # fi
#       done
# done



# python run_experiment.py --mode train --experiment_name test --wandb_project IP --wandb_group training --wandb_name test --dynamics_class InvertedPendulum --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --pretrain --pretrain_iters 1000 --deepReach_model exact 

# base_name="VertDrone"
# # models="exact_diff exact exact_sin exact_exp "
# models="reg"
# for model in $models; do
#     for j in $(seq 1 3);
#       do
#           exp_name="${base_name}_${model}_seed_${j}_pretrain"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "VertDrone_${model}" --wandb_group training --wandb_name ${exp_name} --dynamics_class VertDrone2D --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --pretrain --pretrain_iters 10000 --deepReach_model ${model} --seed $j --input_magnitude_max 1 --K 12 --gravity 9.8
#           wait
#           sleep 50
#           # if [ ${model} != "reg" ]; then
#           #   wait
#           #   exp_name="${base_name}_${model}_seed_${j}_nopretrain"
#           #   python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project IP --wandb_group training --wandb_name ${exp_name} --dynamics_class InvertedPendulum --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --deepReach_model ${model} --seed $j
#           #   wait
#           #   sleep 100
#           # fi
#       done
# done


# python run_experiment.py --mode train --experiment_name test --wandb_project vertdrone --wandb_group training --wandb_name test --dynamics_class VertDrone2D --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --pretrain --pretrain_iters 1000 --deepReach_model reg --input_magnitude_max 1 --K 12 --gravity 9.8
          




# python run_experiment.py --mode test --experiment_dir ./runs --experiment_name quadrotor --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery


# base_name="Quadrotor"
# # models="exact_diff exact exact_sin exact_exp "
# models="exact reg"
# for model in $models; do
#     for j in $(seq 0 0);
#       do
#           exp_name="${base_name}_${model}_seed_${j}_pretrain"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "${base_name}" --wandb_group "${model}_factor" --wandb_name ${exp_name} --experiment_class DeepReach --dynamics_class Quadrotor --pretrain --pretrain_iters 10000 --counter_end 150000 --num_epochs 170000 --num_src_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --collisionR 0.4 --collective_thrust_max 20.0 --set_mode avoid --deepReach_model ${model} --seed ${j}
#           wait
#           sleep 10 #wait for wandb to finish
#           # if [ ${model} != "reg" ]; then
#           #   wait
#           #   exp_name="${base_name}_${model}_seed_${j}_nopretrain"
#           #   python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project IP --wandb_group training --wandb_name ${exp_name} --dynamics_class InvertedPendulum --minWith target --tMax 2 --num_epochs 120000 --counter_end 100000 --num_nl 128 --lr 2e-5 --deepReach_model ${model} --seed $j
#           #   wait
#           #   sleep 100
#           # fi
#       done
# done


# base_name="QuadrotorReachAvoid"
# # models="exact_diff exact exact_sin exact_exp "
# models="exact reg"
# for model in $models; do
#     for j in $(seq 0 0);
#       do
#           exp_name="${base_name}_${model}_seed_${j}_pretrain"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "${base_name}" --wandb_group ${model} --wandb_name ${exp_name} --experiment_class DeepReach --dynamics_class $base_name --pretrain --pretrain_iters 10000 --counter_end 150000 --num_epochs 170000 --num_src_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --collisionR 0.4 --collective_thrust_max 15.0 --deepReach_model ${model} --seed ${j}
#           wait
#           sleep 10 #wait for wandb to finish
#       done
# done




base_name="Dubins3DReachAvoid2"
models="exact"
for model in $models; do
    for j in $(seq 0 4);
      do
          # exp_name="${base_name}_${model}_seed_${j}_pretrain"
          # wait
          # python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "Dubins3DReachAvoid2" --wandb_group ${model} --wandb_name ${exp_name} --dynamics_class Dubins3DReachAvoid --minWith target --tMax 4  --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 100000 --num_nl 512 --lr 2e-5 --deepReach_model ${model} --seed $j --numpoints 10000 --num_src_samples 1000
          # wait
          # sleep 50
          exp_name="${base_name}_${model}_seed_${j}"
          python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "Dubins3DReachAvoid" --wandb_group "exact_nopretrain" --wandb_name ${exp_name} --dynamics_class Dubins3DReachAvoid --minWith target --tMax 4  --num_epochs 110000 --counter_end 100000 --num_nl 512 --lr 2e-5 --deepReach_model ${model} --seed $j --numpoints 10000 --num_src_samples 1000
          wait
          sleep 50
      done
done


# base_name="Dubins3DAvoid"
# models="reg"
# for model in $models; do
#     for j in $(seq 0 0);
#       do
#           exp_name="${base_name}_${model}_seed_${j}_pretrain"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project "Dubins3DAvoid" --wandb_group ${model} --wandb_name ${exp_name} --dynamics_class Dubins3DAvoid --minWith target --tMax 2  --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 100000 --num_nl 512 --lr 2e-5 --deepReach_model ${model} --seed $j 
#           wait
#           sleep 50
#       done
# done


# base_name="DroneDelivery"
# models="reg"
# for model in $models; do
#     for j in $(seq 0 2);
#       do
#           exp_name="${base_name}_${model}_seed_${j}_pretrain2"
#           wait
#           python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project $base_name --wandb_name ${exp_name}  --wandb_group ${model}  --dynamics_class $base_name --minWith target --tMax 4  --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 100000 --num_nl 512 --lr 2e-5 --deepReach_model ${model} --seed $j
#           wait
#           sleep 50
#         #   exp_name="${base_name}_${model}_seed_${j}"
#         #   python run_experiment.py --mode train --experiment_name ${exp_name} --wandb_project $base_name --wandb_name ${exp_name}  --wandb_group ${model}  --dynamics_class $base_name --minWith target --tMax 4  --num_epochs 110000 --counter_end 100000 --num_nl 512 --lr 2e-5 --deepReach_model ${model} --seed $j
#         #   wait
#         #   sleep 50
#       done
# done

# python run_experiment.py --mode train --experiment_name Quadrotor_exact_seed_0_pretrain --wandb_project Quadrotor --wandb_group vanilla_discrete --wandb_name Quadrotor_exact_seed_0_pretrain --experiment_class DeepReach --dynamics_class QuadrotorDiscrete --counter_end 100000 --num_epochs 110000 --num_src_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --collisionR 0.4 --collective_thrust_max 20.0 --deepReach_model exact --seed 0

python run_experiment.py --mode train --experiment_name RimlessWheel_exact2 --wandb_project ExactBC --wandb_group RimlessWheel --wandb_name RimlessWheel_exact2 --experiment_class DeepReach --dynamics_class RimlessWheel --pretrain --pretrain_iters 10000 --counter_end 100000 --num_epochs 120000 --num_switch_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --deepReach_model exact --seed 0 --num_nl 128 --tMax 12 --val_time_resolution 6

python run_experiment.py --mode train --experiment_name compass_walker1 --wandb_project ExactBC --wandb_group CompassWalker --wandb_name compass_walker1 --experiment_class DeepReach --dynamics_class CompassWalker --pretrain --pretrain_iters 10000 --counter_end 100000 --num_epochs 120000 --num_switch_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --deepReach_model exact --seed 0 --num_nl 128 --tMax 1.5 --val_time_resolution 6

python run_experiment.py --mode train --experiment_name compass_walker1 --wandb_project ExactBC --wandb_group CompassWalker --wandb_name compass_walker1 --experiment_class DeepReach --dynamics_class CompassWalker --counter_end 100000 --num_epochs 120000 --num_switch_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --deepReach_model exact --seed 0 --num_nl 128 --tMax 1.5 --val_time_resolution 6

python run_experiment.py --mode train --experiment_name compass_walker1e1 --wandb_project ExactBC --wandb_group CompassWalker --wandb_name compass_walker1e1 --experiment_class DeepReach --dynamics_class CompassWalker --pretrain --pretrain_iters 5000 --counter_end 50000 --num_epochs 55000 --num_switch_samples 12000 --minWith target --dirichlet_loss_divisor 0.023 --batch_size 32 --lr 0.00002 --deepReach_model exact --seed 0 --num_nl 256 --tMax 1 --val_time_resolution 6
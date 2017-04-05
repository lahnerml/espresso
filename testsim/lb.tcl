set base_level 1
set max_level 2

set final_time 10.0
set force 0.01

set do_refine 0
set with_wall 0
set iter 100

cellsystem domain_decomposition -no_verlet_list
setmd skin 0.1
inter 0 0 lennard-jones 1.0 0.1 0.8 auto 0.0
setmd box_l 4 4 1

set agrid [expr 1./pow(2, $max_level)]
set dt [expr $final_time/$iter]

set dt_real $dt
set iter_real $iter

thermostat lb 0
setmd time_step $dt_real

lbadapt-init $base_level
lbadapt_set_max_level $max_level

lbfluid cpu agrid $agrid dens 1 visc 0.05 tau $dt_real friction 0.5 ext_force $force 0 0

if {$with_wall >= 1} {
  lbboundary wall dist  0.5 normal 0  1 0
  lbboundary wall dist -3.5 normal 0 -1 0
}

if {$do_refine >= 1} {
  lbadapt-regref 0 2 0 2 0 0.5
}

lbfluid print vtk velocity lb_base_${base_level}_max_${max_level}_wall_${with_wall}_refine_${do_refine}_pre.vtk

lbadapt-reset-fluid
integrate $iter_real
puts "time=[setmd time] u_mid=[ lindex [ lbnode 1 [expr int(2.0/$agrid)] 1 print u ] 0 ]"

lbfluid print vtk velocity lb_base_${base_level}_max_${max_level}_wall_${with_wall}_refine_${do_refine}_post.vtk

exit

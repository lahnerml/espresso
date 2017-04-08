set base_level 1
set max_level 2

set final_time 1000.0
set force 0.001

set do_refine 1
set with_wall 1
set iter 4000

cellsystem domain_decomposition -no_verlet_list
setmd skin 0.1
inter 0 0 lennard-jones 1.0 0.1 0.8 auto 0.0
setmd box_l 4 12 1

set agrid [expr 1./pow(2, $max_level)]
set dt [expr $final_time/$iter]

set dt_real [expr $dt/pow(2, $max_level - 1)]
set iter_real [expr int($iter*pow(2, $max_level - 1))]

thermostat lb 0
setmd time_step $dt_real

lbadapt-init $base_level
lbadapt_set_max_level $max_level

lbfluid cpu agrid $agrid dens 1 visc 0.05 tau $dt_real friction 0.5 ext_force 0 0 0

if {$with_wall >= 1} {
  lbboundary wall dist  0.5 normal 0  1 0 ;#velocity  0.1 0 0
  lbboundary wall dist -11.5 normal 0 -1 0 ;#velocity -0.1 0 0
  
  lbboundary rhomboid corner -0.5 5.5 -0.5 a 5 0 0 b 0 1 0 c 0 0 2 direction outside velocity 0.1 0 0
}

if {$do_refine >= 1} {
  #lbadapt-geom-ref
  #lbadapt-regref 0 4 4.5 7.5 0 1
  lbadapt-regref 0 4 8 12 0 1
  lbadapt-regref 0 4 0 4 0 1
}

lbfluid print vtk velocity lb_${base_level}_${max_level}_0.vtk

lbadapt-reset-fluid
integrate $iter_real
puts "time=[setmd time] u_mid=[ lindex [ lbnode 1 [expr int(2.0/$agrid)] 1 print u ] 0 ]"

lbfluid print vtk velocity "lb_${base_level}_${max_level}_1.vtk"

exit
